"""
Payment service for TPC-C operations
"""

import logging
from typing import Any, Dict, List, Optional

from database.base_connector import BaseDatabaseConnector

logger = logging.getLogger(__name__)


class PaymentService:
    """Service class for payment-related operations"""

    def __init__(self, db_connector: BaseDatabaseConnector):
        self.db = db_connector

    def execute_payment(
        self, warehouse_id: int, district_id: int, customer_id: int, amount: float
    ) -> Dict[str, Any]:
        """Execute TPC-C Payment transaction"""
        try:
            return self.db.execute_payment(
                warehouse_id, district_id, customer_id, amount
            )
        except Exception as e:
            logger.error(f"Payment service error: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_payment_history(
        self,
        warehouse_id: Optional[int] = None,
        district_id: Optional[int] = None,
        customer_id: Optional[int] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get payment history with optional filters"""
        try:
            return self.db.get_payment_history(
                warehouse_id, district_id, customer_id, limit
            )
        except Exception as e:
            logger.error(f"Get payment history service error: {str(e)}")
            return []

    def get_payment_history_paginated(
        self,
        warehouse_id: Optional[int] = None,
        district_id: Optional[int] = None,
        customer_id: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get payment history with pagination"""
        try:
            return self.db.get_payment_history_paginated(
                warehouse_id, district_id, customer_id, limit, offset
            )
        except Exception as e:
            logger.error(f"Get payment history paginated service error: {str(e)}")
            return {
                "payments": [],
                "total_count": 0,
                "limit": limit,
                "offset": offset,
                "has_next": False,
                "has_prev": False,
            }

    def get_customer_payment_summary(
        self, warehouse_id: int, district_id: int, customer_id: int
    ) -> Dict[str, Any]:
        """Get payment summary for a specific customer"""
        try:
            # Get customer info and payment stats
            customer_query = """
                SELECT c.c_first, c.c_middle, c.c_last, c.c_balance,
                       c.c_ytd_payment, c.c_payment_cnt, c.c_credit,
                       c.c_credit_lim, c.c_discount, c.c_since
                FROM customer c
                WHERE c.c_w_id = %s AND c.c_d_id = %s AND c.c_id = %s
            """

            customer_result = self.db.execute_query(
                customer_query, (warehouse_id, district_id, customer_id)
            )

            if not customer_result:
                return {"success": False, "error": "Customer not found"}

            customer = customer_result[0]

            # Get recent payment history
            history_query = """
                SELECT h.h_date, h.h_amount, h.h_data
                FROM history h
                WHERE h.h_c_w_id = %s AND h.h_c_d_id = %s AND h.h_c_id = %s
                ORDER BY h.h_date DESC
                LIMIT 10
            """

            payment_history = self.db.execute_query(
                history_query, (warehouse_id, district_id, customer_id)
            )

            # Calculate payment statistics
            stats_query = """
                SELECT 
                    COUNT(*) as total_payments,
                    SUM(h_amount) as total_amount,
                    AVG(h_amount) as avg_amount,
                    MIN(h_amount) as min_amount,
                    MAX(h_amount) as max_amount,
                    MIN(h_date) as first_payment,
                    MAX(h_date) as last_payment
                FROM history h
                WHERE h.h_c_w_id = %s AND h.h_c_d_id = %s AND h.h_c_id = %s
            """

            stats_result = self.db.execute_query(
                stats_query, (warehouse_id, district_id, customer_id)
            )

            payment_stats = stats_result[0] if stats_result else {}

            return {
                "success": True,
                "customer": customer,
                "payment_history": payment_history,
                "payment_stats": payment_stats,
            }

        except Exception as e:
            logger.error(f"Get customer payment summary service error: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_payment_statistics(
        self, warehouse_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get payment statistics"""
        try:
            stats = {}

            # Base query conditions
            where_clause = "WHERE 1=1"
            params = []

            if warehouse_id:
                where_clause += " AND h_w_id = %s"
                params.append(warehouse_id)

            # Total payments
            total_query = f"SELECT COUNT(*) as count FROM history {where_clause}"
            total_result = self.db.execute_query(total_query, tuple(params))
            stats["total_payments"] = total_result[0]["count"] if total_result else 0

            # Total payment amount
            amount_query = f"SELECT SUM(h_amount) as total FROM history {where_clause}"
            amount_result = self.db.execute_query(amount_query, tuple(params))
            stats["total_payment_amount"] = (
                float(amount_result[0]["total"])
                if amount_result and amount_result[0]["total"]
                else 0.0
            )

            # Average payment amount
            avg_query = f"SELECT AVG(h_amount) as avg FROM history {where_clause}"
            avg_result = self.db.execute_query(avg_query, tuple(params))
            stats["avg_payment_amount"] = (
                float(avg_result[0]["avg"])
                if avg_result and avg_result[0]["avg"]
                else 0.0
            )

            # Payments today
            today_query = f"""
                SELECT COUNT(*) as count, COALESCE(SUM(h_amount), 0) as amount
                FROM history 
                {where_clause} AND DATE(h_date) = CURRENT_DATE
            """
            today_result = self.db.execute_query(today_query, tuple(params))
            if today_result:
                stats["payments_today"] = today_result[0]["count"]
                stats["payment_amount_today"] = float(today_result[0]["amount"])
            else:
                stats["payments_today"] = 0
                stats["payment_amount_today"] = 0.0

            # Top customers by payment amount
            top_customers_query = f"""
                SELECT c.c_id, c.c_w_id, c.c_d_id, c.c_first, c.c_middle, c.c_last,
                       c.c_ytd_payment, c.c_payment_cnt
                FROM customer c
                {where_clause.replace("h_w_id", "c.c_w_id")}
                ORDER BY c.c_ytd_payment DESC
                LIMIT 5
            """
            top_customers_result = self.db.execute_query(
                top_customers_query, tuple(params)
            )
            stats["top_customers"] = top_customers_result

            return stats

        except Exception as e:
            logger.error(f"Get payment statistics service error: {str(e)}")
            return {}

    def get_recent_payments(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get most recent payments across all warehouses"""
        try:
            query = """
                SELECT h.h_date, h.h_amount, h.h_data,
                       h.h_c_id, h.h_c_w_id, h.h_c_d_id,
                       c.c_first, c.c_middle, c.c_last,
                       w.w_name
                FROM history h
                JOIN customer c ON c.c_w_id = h.h_c_w_id AND c.c_d_id = h.h_c_d_id AND c.c_id = h.h_c_id
                JOIN warehouse w ON w.w_id = h.h_c_w_id
                ORDER BY h.h_date DESC
                LIMIT %s
            """

            return self.db.execute_query(query, (limit,))

        except Exception as e:
            logger.error(f"Get recent payments service error: {str(e)}")
            return []

    def get_payment_trends(
        self, warehouse_id: Optional[int] = None, days: int = 30
    ) -> Dict[str, Any]:
        """Get payment trends over specified number of days"""
        try:
            # Base query conditions
            where_clause = "WHERE h_date >= CURRENT_DATE - INTERVAL '%s days'"
            params = [days]

            if warehouse_id:
                where_clause += " AND h_w_id = %s"
                params.append(warehouse_id)

            # Daily payment trends
            daily_query = f"""
                SELECT DATE(h_date) as payment_date,
                       COUNT(*) as payment_count,
                       SUM(h_amount) as total_amount,
                       AVG(h_amount) as avg_amount
                FROM history
                {where_clause}
                GROUP BY DATE(h_date)
                ORDER BY payment_date DESC
            """

            daily_trends = self.db.execute_query(daily_query, tuple(params))

            # Payment amount distribution
            distribution_query = f"""
                SELECT 
                    COUNT(CASE WHEN h_amount < 100 THEN 1 END) as under_100,
                    COUNT(CASE WHEN h_amount >= 100 AND h_amount < 500 THEN 1 END) as between_100_500,
                    COUNT(CASE WHEN h_amount >= 500 AND h_amount < 1000 THEN 1 END) as between_500_1000,
                    COUNT(CASE WHEN h_amount >= 1000 THEN 1 END) as over_1000
                FROM history
                {where_clause}
            """

            distribution_result = self.db.execute_query(
                distribution_query, tuple(params)
            )
            distribution = distribution_result[0] if distribution_result else {}

            return {
                "success": True,
                "daily_trends": daily_trends,
                "amount_distribution": distribution,
                "period_days": days,
            }

        except Exception as e:
            logger.error(f"Get payment trends service error: {str(e)}")
            return {"success": False, "error": str(e)}

    def validate_payment_data(
        self, warehouse_id: int, district_id: int, customer_id: int, amount: float
    ) -> Dict[str, Any]:
        """Validate payment data before processing"""
        try:
            errors = []

            # Validate amount
            if amount <= 0:
                errors.append("Payment amount must be positive")

            if amount > 10000:  # Arbitrary large amount check
                errors.append("Payment amount exceeds maximum allowed")

            # Check if customer exists
            customer_query = """
                SELECT c_id, c_first, c_last, c_balance, c_credit_lim
                FROM customer
                WHERE c_w_id = %s AND c_d_id = %s AND c_id = %s
            """

            customer_result = self.db.execute_query(
                customer_query, (warehouse_id, district_id, customer_id)
            )

            if not customer_result:
                errors.append("Customer not found")
            else:
                customer = customer_result[0]
                # Check if payment would exceed credit limit (if applicable)
                new_balance = customer["c_balance"] - amount
                if new_balance < -customer["c_credit_lim"]:
                    errors.append("Payment would exceed customer credit limit")

            # Check if warehouse and district exist
            district_query = """
                SELECT d_id, w.w_name
                FROM district d
                JOIN warehouse w ON w.w_id = d.d_w_id
                WHERE d.d_w_id = %s AND d.d_id = %s
            """

            district_result = self.db.execute_query(
                district_query, (warehouse_id, district_id)
            )

            if not district_result:
                errors.append("Warehouse or district not found")

            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "customer": customer_result[0] if customer_result else None,
                "district": district_result[0] if district_result else None,
            }

        except Exception as e:
            logger.error(f"Payment validation service error: {str(e)}")
            return {"valid": False, "errors": [str(e)]}
