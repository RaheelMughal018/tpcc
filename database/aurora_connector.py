import logging
import os
import time
from typing import Any, Dict, List, Optional
from datetime import datetime
import boto3
import psycopg2
from psycopg2.extras import RealDictCursor
from botocore.config import Config

from .base_connector import BaseDatabaseConnector

logger = logging.getLogger(__name__)


class AuroraDSQLConnector(BaseDatabaseConnector):
    """
    Amazon Aurora DSQL database connector for TPC-C application
    """

    def __init__(self):
        super().__init__()
        self.provider_name = "Amazon Aurora DSQL"
        self.connection = None

        # ---- config ----
        self.region = os.getenv("AWS_REGION")
        self.cluster_endpoint = os.getenv("DSQL_CLUSTER_ENDPOINT")
        self.db_name = os.getenv("DSQL_DB", "postgres")
        self.db_user = os.getenv("DSQL_USER", "admin")
        self.db_port = int(os.getenv("DSQL_PORT", "5432"))
        self.ssl_mode = os.getenv("DSQL_SSLMODE", "require")
        self.use_iam = os.getenv("DSQL_USE_IAM", "true").lower() == "true"
        self.token_ttl = int(os.getenv("DSQL_TOKEN_TTL", "900"))  # seconds; default 15m

        if not all([self.region, self.cluster_endpoint]):
            raise ValueError(
                "Missing required Aurora DSQL configuration: AWS_REGION and DSQL_CLUSTER_ENDPOINT"
            )

        # DSQL client (for token generation)
        self._dsql = (
            boto3.client("dsql", region_name=self.region) if self.use_iam else None
        )

        # (optional) log which AWS principal we’re using — helps catch profile mismatches
        try:
            sts = boto3.client("sts", config=Config(region_name=self.region))
            ident = sts.get_caller_identity()
            logger.info(f"AuroraDSQLConnector using AWS principal: {ident.get('Arn')}")
        except Exception:
            logger.info("Could not determine AWS caller identity (non-fatal).")

        # initial connect
        self._connect()

    # ---------------- internal helpers ----------------

    def _normalize_token(self, token: str) -> str:
        """Aurora DSQL token may be a presigned URL; strip leading scheme if present."""
        if token.startswith("https://"):
            return token[len("https://") :]
        return token

    def _generate_token(self) -> str:
        """
        Generate a fresh Aurora DSQL auth token (valid for token_ttl seconds).
        Uses DbConnectAdmin for 'admin' user, else DbConnect.
        """
        if not self.use_iam:
            pwd = os.getenv("DSQL_PASSWORD")
            if not pwd:
                raise ValueError("DSQL_PASSWORD not set and DSQL_USE_IAM=false")
            return pwd

        if self._dsql is None:
            self._dsql = boto3.client("dsql", region_name=self.region)

        # Choose the proper action for the DB role
        if self.db_user == "admin":
            # generate_db_connect_admin_auth_token(hostname, region, expires)
            raw = self._dsql.generate_db_connect_admin_auth_token(
                Hostname=self.cluster_endpoint,
                Region=self.region,
                ExpiresIn=self.token_ttl,
            )
        else:
            raw = self._dsql.generate_db_connect_auth_token(
                Hostname=self.cluster_endpoint,
                Region=self.region,
                ExpiresIn=self.token_ttl,
            )

        token = self._normalize_token(raw)
        return token

    def _connect(self, force_new_token: bool = True):
        """(Re)establish a connection. Always uses a FRESH DSQL token when IAM is enabled."""
        # If we already have an open connection and not forcing new token, reuse
        if (
            self.connection
            and not getattr(self.connection, "closed", True)
            and not force_new_token
        ):
            return

        if self.connection and not self.connection.closed:
            try:
                self.connection.close()
            except Exception:
                pass

        password = self._generate_token()

        try:
            self.connection = psycopg2.connect(
                host=self.cluster_endpoint,
                port=self.db_port,
                database=self.db_name,
                user=self.db_user,
                password=password,
                sslmode=self.ssl_mode,  # 'require' is OK; use 'verify-full' + sslrootcert in prod
            )
            self.connection.autocommit = True
            logger.info("Successfully (re)connected to Aurora DSQL")
        except Exception as e:
            logger.error(f"Failed to connect to Aurora DSQL: {e}")
            raise

    def _ensure_connection(self):
        """Open or refresh connection if needed."""
        if not self.connection or getattr(self.connection, "closed", True):
            self._connect(force_new_token=True)

    # ---------------- public API ----------------

    def test_connection(self) -> bool:
        """Return True if SELECT 1 succeeds. Retry once with fresh token on IAM failures."""
        try:
            self._ensure_connection()
            with self.connection.cursor() as cur:
                cur.execute("SELECT 1")
                row = cur.fetchone()
                ok = bool(row and row[0] == 1)
                logger.info(
                    f"Aurora DSQL connection test {'successful' if ok else 'failed'}"
                )
                return ok
        except psycopg2.OperationalError as e:
            msg = str(e).lower()
            if self.use_iam and (
                "access denied" in msg or "expired" in msg or "authentication" in msg
            ):
                logger.warning(
                    "Auth error during test_connection; refreshing token and retrying once..."
                )
                self._connect(force_new_token=True)
                try:
                    with self.connection.cursor() as cur:
                        cur.execute("SELECT 1")
                        row = cur.fetchone()
                        return bool(row and row[0] == 1)
                except Exception as e2:
                    logger.error(f"Retry test_connection failed: {e2}")
                    return False
            logger.error(f"Aurora DSQL connection test failed: {e}")
            return False
        except Exception as e:
            logger.error(f"Aurora DSQL connection test failed: {e}")
            return False

    def execute_query(
        self, query: str, params: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a SQL statement.
        - Returns list[dict] for SELECT.
        - For non-SELECT, returns [{'rowcount': N}] so callers don't crash on fetchall().
        - On IAM auth errors, refreshes token and retries once.
        """
        start_time = time.time()
        self._ensure_connection()

        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cur:
                if params:
                    cur.execute(query, params)
                else:
                    cur.execute(query)

                # cur.description is None for statements that don't return rows
                if cur.description is None:
                    rc = cur.rowcount
                    exec_time = time.time() - start_time
                    logger.info(
                        f"Non-SELECT executed in {exec_time:.2f}s (rowcount={rc})"
                    )
                    return [{"rowcount": rc}]

                rows = cur.fetchall()
                exec_time = time.time() - start_time
                logger.info(f"Query executed in {exec_time:.2f}s (rows={len(rows)})")
                return [dict(r) for r in rows]

        except psycopg2.OperationalError as e:
            # Handle token expiry / access denied by refreshing once
            msg = str(e).lower()
            if self.use_iam and (
                "access denied" in msg or "expired" in msg or "authentication" in msg
            ):
                logger.warning(
                    "OperationalError (likely token); refreshing token and retrying once..."
                )
                try:
                    self._connect(force_new_token=True)
                    with self.connection.cursor(cursor_factory=RealDictCursor) as cur:
                        if params:
                            cur.execute(query, params)
                        else:
                            cur.execute(query)
                        if cur.description is None:
                            return [{"rowcount": cur.rowcount}]
                        return [dict(r) for r in cur.fetchall()]
                except Exception as e2:
                    if self.connection:
                        try:
                            self.connection.rollback()
                        except Exception:
                            pass
                    logger.error(f"Retry after token refresh failed: {e2}")
                    raise
            # other operational errors
            if self.connection:
                try:
                    self.connection.rollback()
                except Exception:
                    pass
            logger.error(f"Aurora DSQL query execution failed: {e}")
            raise

        except Exception as e:
            if self.connection:
                try:
                    self.connection.rollback()
                except Exception:
                    pass
            logger.error(f"Aurora DSQL query execution failed: {e}")
            raise

    def get_provider_name(self) -> str:
        return self.provider_name

    def close_connection(self):
        try:
            if self.connection and not getattr(self.connection, "closed", True):
                self.connection.close()
                logger.info("Aurora DSQL connection closed successfully")
        except Exception as e:
            logger.error(f"Connection cleanup failed: {e}")
    def get_warehouses(self):
        query = 'SELECT * FROM warehouse'
        return self.execute_query(query)
    
    def execute_new_order(self, warehouse_id: int, district_id: int, customer_id: int, items: list) -> Dict[str, Any]:
        """
        Create a new order, insert order lines, update stock, and add entry to new_order table.
        Uses direct SQL queries with psycopg2 instead of ORM.

        Args:
            warehouse_id (int): Warehouse ID
            district_id (int): District ID
            customer_id (int): Customer ID
            items (list): List of {"item_id": int, "quantity": int}
        Returns:
            dict: {"success": True, "order_id": ...} or {"success": False, "error": ...}
        """
        self._ensure_connection()
        
        try:
            # Start transaction
            with self.connection:  # This automatically handles commit/rollback
                with self.connection.cursor() as cur:
                    # 1️⃣ Generate new order_id
                    cur.execute('SELECT COALESCE(MAX(o_id), 0) + 1 FROM "order"')
                    order_id = cur.fetchone()[0]

                    entry_d = datetime.now()
                    carrier_id = None
                    ol_cnt = len(items)

                    # 2️⃣ Insert into orders table
                    cur.execute("""
                        INSERT INTO "order" (o_id, o_w_id, o_d_id, o_c_id, o_entry_d, o_carrier_id, o_ol_cnt, o_all_local)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """, (
                        order_id, warehouse_id, district_id, customer_id,
                        entry_d, carrier_id, ol_cnt, 1  # 1 for all_local
                    ))

                    # 3️⃣ Insert order lines and update stock
                    line_number = 1
                    for item in items:
                        i_id = item.get("item_id") or item.get("i_id")
                        qty = item["quantity"]

                        # Fetch stock quantity and item price
                        cur.execute("""
                            SELECT s_quantity, i_price 
                            FROM stock 
                            JOIN item ON stock.s_i_id = item.i_id
                            WHERE s_i_id = %s AND s_w_id = %s
                        """, (i_id, warehouse_id))
                        
                        stock_row = cur.fetchone()
                        if not stock_row:
                            raise ValueError(f"Item {i_id} not found in warehouse {warehouse_id}")
                        
                        s_quantity, i_price = stock_row
                        if s_quantity < qty:
                            raise ValueError(f"Not enough stock for item {i_id}")

                        amount = qty * i_price

                        # Insert into order_line
                        cur.execute("""
                            INSERT INTO order_line 
                            (ol_o_id, ol_w_id, ol_d_id, ol_number, ol_i_id, ol_supply_w_id, 
                             ol_quantity, ol_amount, ol_dist_info)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """, (
                            order_id, warehouse_id, district_id, line_number,
                            i_id, warehouse_id, qty, amount, "SAMPLE_DIST_INFO"
                        ))

                        # Update stock quantity
                        cur.execute("""
                            UPDATE stock
                            SET s_quantity = s_quantity - %s
                            WHERE s_w_id = %s AND s_i_id = %s
                        """, (qty, warehouse_id, i_id))

                        line_number += 1

                    # 4️⃣ Insert into new_order table
                    cur.execute("""
                        INSERT INTO new_order (no_o_id, no_d_id, no_w_id)
                        VALUES (%s, %s, %s)
                    """, (order_id, district_id, warehouse_id))

            return {"success": True, "order_id": order_id}

        except Exception as e:
            logger.error(f"Failed to execute new order: {str(e)}")
            return {"success": False, "error": str(e)}
        
    def execute_payment(
        self,
        warehouse_id: int,
        district_id: int,
        customer_id: int,
        amount: float
    ) -> Dict[str, Any]:
        """
        Execute TPC-C Payment transaction against Aurora DSQL using raw SQL.
        Runs as a single transaction; returns the customer's new balance.

        NOTE: TPC-C typically DECREASES c_balance by amount (payment).
            If in your schema c_balance means "cash on hand" (not debt), flip the sign below.
        """
        self._ensure_connection()

        try:
            with self.connection:              # begin/commit (auto-rollback on error)
                with self.connection.cursor() as cur:
                    # 1) Update warehouse & district YTD (optional but standard TPC-C)
                    cur.execute(
                        "UPDATE warehouse SET w_ytd = COALESCE(w_ytd, 0) + %s WHERE w_id = %s",
                        (amount, warehouse_id),
                    )
                    cur.execute(
                        """
                        UPDATE district
                        SET d_ytd = COALESCE(d_ytd, 0) + %s
                        WHERE d_w_id = %s AND d_id = %s
                        """,
                        (amount, warehouse_id, district_id),
                    )

                    # 2) Update customer balance and get the new balance
                    #    TPC-C: c_balance := c_balance - amount
                    cur.execute(
                        """
                        UPDATE customer
                        SET c_balance = c_balance - %s
                        WHERE c_w_id = %s AND c_d_id = %s AND c_id = %s
                        RETURNING c_balance
                        """,
                        (amount, warehouse_id, district_id, customer_id),
                    )
                    row = cur.fetchone()
                    if not row:
                        raise ValueError(
                            f"Customer not found (w_id={warehouse_id}, d_id={district_id}, c_id={customer_id})"
                        )
                    new_balance = float(row[0])

                    # 3) Insert history row (must include h_d_id and h_w_id)
                    cur.execute(
                        """
                        INSERT INTO history (
                            h_c_id, h_c_d_id, h_c_w_id,
                            h_d_id, h_w_id,
                            h_date, h_amount, h_data
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            customer_id,          # H_C_ID
                            district_id,          # H_C_D_ID
                            warehouse_id,         # H_C_W_ID
                            district_id,          # H_D_ID (payment district)
                            warehouse_id,         # H_W_ID (payment warehouse)
                            datetime.now(),       # H_DATE
                            amount,               # H_AMOUNT
                            "PAYMENT",            # H_DATA (24 chars max in classic TPC-C)
                        ),
                    )

            return {
                "success": True,
                "customer_id": customer_id,
                "amount": amount,
                "new_balance": new_balance,
            }

        except Exception as e:
            logger.error(f"Payment failed: {e}")
            return {"success": False, "error": str(e)} 

    def get_stock_level(self, warehouse_id: int, district_id: int, threshold: int) -> Dict[str, Any]:
        """
        Execute TPC-C Stock Level transaction against Aurora DSQL:
        Returns the count of items in a district's warehouse with stock below a threshold.

        Args:
            warehouse_id (int): Warehouse ID
            district_id (int): District ID
            threshold (int): Stock threshold

        Returns:
            dict: {"success": True, "low_stock_count": int} or {"success": False, "error": ...}
        """
        self._ensure_connection()

        try:
            with self.connection:  # auto commit/rollback
                with self.connection.cursor() as cur:
                    cur.execute(
                        """
                        SELECT COUNT(*) AS low_stock_count
                        FROM stock s    
                        JOIN district d
                            ON d.d_w_id = s.s_w_id
                        AND d.d_id = %s
                        WHERE s.s_w_id = %s
                        AND s.s_quantity < %s
                        """,
                        (district_id, warehouse_id, threshold),
                    )
                    row = cur.fetchone()
                    low_stock_count = int(row[0]) if row else 0

            return {"success": True, "low_stock_count": low_stock_count}

        except Exception as e:
            logger.error(f"Failed to get stock level: {e}")
            return {"success": False, "error": str(e)}
