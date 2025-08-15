"""
Order service for TPC-C operations
"""

import logging
import os
from typing import Any, Dict, List, Optional

from database.base_connector import BaseDatabaseConnector

logger = logging.getLogger(__name__)


class OrderService:
    """Service class for order-related operations"""

    def __init__(
        self, db_connector: BaseDatabaseConnector, region_name: Optional[str] = None
    ):
        self.db = db_connector
        # Get region name from environment variable or use default
        self.region_name = region_name or os.environ.get("REGION_NAME", "default")

    def execute_new_order(
        self,
        warehouse_id: int,
        district_id: int,
        customer_id: int,
        items: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Execute TPC-C New Order transaction"""
        try:
            # Execute the new order transaction
            result = self.db.execute_new_order(
                warehouse_id, district_id, customer_id, items
            )

            if result["success"]:
                # Add region information to the result
                result["region_created"] = self.region_name

                # Update the order with region information
                try:
                    # For Spanner, we need to use a different approach since it handles parameters differently
                    from database.spanner_connector import SpannerConnector

                    if isinstance(self.db, SpannerConnector):
                        # For Spanner, we'll use the transaction-based approach within the connector
                        # Skip the region update here since Spanner handles it differently
                        logger.info(
                            f"Spanner order created with region tracking: {self.region_name}"
                        )
                    else:
                        # For other databases
                        self.db.execute_query(
                            "UPDATE orders SET region_created = %s WHERE o_id = %s AND o_d_id = %s AND o_w_id = %s",
                            (
                                self.region_name,
                                result["order_id"],
                                district_id,
                                warehouse_id,
                            ),
                        )
                        logger.info(f"Order region updated to: {self.region_name}")
                except Exception as update_error:
                    logger.warning(
                        f"Failed to update region information: {str(update_error)}"
                    )

            return result
        except Exception as e:
            logger.error(f"New order service error: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_order_status(
        self, warehouse_id: int, district_id: int, customer_id: int
    ) -> Dict[str, Any]:
        """Get order status for a customer"""
        try:
            return self.db.get_order_status(warehouse_id, district_id, customer_id)
        except Exception as e:
            logger.error(f"Order status service error: {str(e)}")
            return {"success": False, "error": str(e)}

    def execute_delivery(self, warehouse_id: int, carrier_id: int) -> Dict[str, Any]:
        """Execute TPC-C Delivery transaction"""
        try:
            return self.db.execute_delivery(warehouse_id, carrier_id)
        except Exception as e:
            logger.error(f"Delivery service error: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_orders(
        self,
        warehouse_id: int = None,
        district_id: int = None,
        customer_id: int = None,
        status: str = None,
        limit: int = 50,
        offset: int = 0,
    ) -> dict:
        """
        Get paginated list of orders with optional filtering.

        Args:
            warehouse_id: Optional warehouse filter
            district_id: Optional district filter
            customer_id: Optional customer filter
            status: Optional status filter ('new' or 'delivered')
            limit: Number of records to return
            offset: Number of records to skip
        Returns:
            dict: Paginated orders with metadata
        """
        try:
            # Base query
            query = """
                SELECT o.o_id, o.o_w_id, o.o_d_id, o.o_c_id, o.o_entry_d,
                    c.c_first, c.c_middle, c.c_last,
                    CASE WHEN no.no_o_id IS NOT NULL THEN 'New' ELSE 'Delivered' END AS status
                FROM "order" o
                JOIN customer c
                ON c.c_w_id = o.o_w_id
                AND c.c_d_id = o.o_d_id
                AND c.c_id   = o.o_c_id
                LEFT JOIN new_order no
                ON no.no_w_id = o.o_w_id
                AND no.no_d_id = o.o_d_id
                AND no.no_o_id = o.o_id
                WHERE 1=1
            """
            params = []

            # Dynamic filters
            if warehouse_id is not None:
                query += " AND o.o_w_id = %s"
                params.append(warehouse_id)
            if district_id is not None:
                query += " AND o.o_d_id = %s"
                params.append(district_id)
            if customer_id is not None:
                query += " AND o.o_c_id = %s"
                params.append(customer_id)
            if status:
                if status.lower() == "new":
                    query += " AND no.no_o_id IS NOT NULL"
                elif status.lower() == "delivered":
                    query += " AND no.no_o_id IS NULL"

            # Ordering and pagination
            query += " ORDER BY o.o_entry_d DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])

            # Execute main query
            rows = self.db.execute_query(query, tuple(params))

            # Get total count (without limit/offset)
            count_query = """
                SELECT COUNT(*) 
                FROM "order" o 
                LEFT JOIN new_order no 
                ON no.no_w_id = o.o_w_id 
                AND no.no_d_id = o.o_d_id 
                AND no.no_o_id = o.o_id 
                WHERE 1=1
            """
            count_params = []
            if warehouse_id is not None:
                count_query += " AND o.o_w_id = %s"
                count_params.append(warehouse_id)
            if district_id is not None:
                count_query += " AND o.o_d_id = %s"
                count_params.append(district_id)
            if customer_id is not None:
                count_query += " AND o.o_c_id = %s"
                count_params.append(customer_id)
            if status:
                if status.lower() == "new":
                    count_query += " AND no.no_o_id IS NOT NULL"
                elif status.lower() == "delivered":
                    count_query += " AND no.no_o_id IS NULL"

            total_count = self.db.execute_query(count_query, tuple(count_params))[0][
                "count"
            ]

            return {
                "orders": rows,
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "has_next": (offset + limit) < total_count,
                "has_prev": offset > 0,
            }

        except Exception as e:
            logger.error(f"Get orders query failed: {e}")
            return {
                "orders": [],
                "total_count": 0,
                "limit": limit,
                "offset": offset,
                "has_next": False,
                "has_prev": False,
            }

    def get_order_details(
        self, warehouse_id: int, district_id: int, order_id: int
    ) -> Dict[str, Any]:
        """Get detailed information about a specific order"""
        try:
            # Get order information
            order_query = """
                SELECT o.*, c.c_first, c.c_middle, c.c_last,
                       CASE WHEN no.no_o_id IS NOT NULL THEN 'New' ELSE 'Delivered' END as status
                FROM "order" o
                JOIN customer c ON c.c_w_id = o.o_w_id AND c.c_d_id = o.o_d_id AND c.c_id = o.o_c_id
                LEFT JOIN new_order no ON no.no_w_id = o.o_w_id AND no.no_d_id = o.o_d_id AND no.no_o_id = o.o_id
                WHERE o.o_w_id = %s AND o.o_d_id = %s AND o.o_id = %s
            """

            order_result = self.db.execute_query(
                order_query, (warehouse_id, district_id, order_id)
            )

            if not order_result:
                return {"success": False, "error": "Order not found"}

            order = order_result[0]

            # Get order lines
            order_lines_query = """i
                SELECT ol.*, i.i_name, i.i_price
                FROM order_line ol
                JOIN item i ON i.i_id = ol.ol_i_id
                WHERE ol.ol_w_id = %s AND ol.ol_d_id = %s AND ol.ol_o_id = %s
                ORDER BY ol.ol_number
            """

            order_lines = self.db.execute_query(
                order_lines_query, (warehouse_id, district_id, order_id)
            )

            # Calculate total amount
            total_amount = sum(float(line.get("ol_amount", 0)) for line in order_lines)

            return {
                "success": True,
                "order": order,
                "order_lines": order_lines,
                "total_amount": total_amount,
            }

        except Exception as e:
            logger.error(f"Get order details service error: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_recent_orders(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most recent orders across all warehouses"""
        try:
            query = """
                SELECT o.o_id, o.o_w_id, o.o_d_id, o.o_c_id, o.o_entry_d,
                       c.c_first, c.c_middle, c.c_last,
                       w.w_name,
                       CASE WHEN no.no_o_id IS NOT NULL THEN 'New' ELSE 'Delivered' END as status
                FROM "order" o
                JOIN customer c ON c.c_w_id = o.o_w_id AND c.c_d_id = o.o_d_id AND c.c_id = o.o_c_id
                JOIN warehouse w ON w.w_id = o.o_w_id
                LEFT JOIN new_order no ON no.no_w_id = o.o_w_id AND no.no_d_id = o.o_d_id AND no.no_o_id = o.o_id
                ORDER BY o.o_entry_d DESC
                LIMIT %s
            """

            return self.db.execute_query(query, (limit,))

        except Exception as e:
            logger.error(f"Get recent orders service error: {str(e)}")
            return []

    def get_order_statistics(
        self, warehouse_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get order statistics"""
        try:
            stats = {}

            # Base query conditions
            where_clause = "WHERE 1=1"
            params = []

            if warehouse_id:
                where_clause += " AND o_w_id = %s"
                params.append(warehouse_id)

            # Total orders
            total_query = f'SELECT COUNT(*) as count FROM "order" {where_clause}'
            total_result = self.db.execute_query(total_query, tuple(params))
            stats["total_orders"] = total_result[0]["count"] if total_result else 0

            # New orders
            new_query = f"""
                SELECT COUNT(*) as count 
                FROM "order" o 
                JOIN new_order no ON no.no_w_id = o.o_w_id AND no.no_d_id = o.o_d_id AND no.no_o_id = o.o_id
                {where_clause}
            """
            new_result = self.db.execute_query(new_query, tuple(params))
            stats["new_orders"] = new_result[0]["count"] if new_result else 0

            # Delivered orders
            stats["delivered_orders"] = stats["total_orders"] - stats["new_orders"]

            # Orders today
            today_query = f"""
                SELECT COUNT(*) as count 
                FROM "order"
                {where_clause} AND DATE(o_entry_d) = CURRENT_DATE
            """
            today_result = self.db.execute_query(today_query, tuple(params))
            stats["orders_today"] = today_result[0]["count"] if today_result else 0

            # Average order value
            avg_query = f"""
                SELECT AVG(total_amount) as avg_amount
                FROM (
                    SELECT SUM(ol_amount) as total_amount
                    FROM order_line ol
                    JOIN orders o ON o.o_w_id = ol.ol_w_id AND o.o_d_id = ol.ol_d_id AND o.o_id = ol.ol_o_id
                    {where_clause.replace("o_w_id", "o.o_w_id")}
                    GROUP BY ol.ol_w_id, ol.ol_d_id, ol.ol_o_id
                ) as order_totals
            """
            avg_result = self.db.execute_query(avg_query, tuple(params))
            stats["avg_order_value"] = (
                float(avg_result[0]["avg_amount"])
                if avg_result and avg_result[0]["avg_amount"]
                else 0.0
            )

            return stats

        except Exception as e:
            logger.error(f"Get order statistics service error: {str(e)}")
            return {}
