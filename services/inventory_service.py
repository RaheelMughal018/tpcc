"""
Inventory service for TPC-C operations
"""

import logging
from typing import Any, Dict, List, Optional

from database.base_connector import BaseDatabaseConnector

logger = logging.getLogger(__name__)


class InventoryService:
    """Service class for inventory-related operations"""

    def __init__(self, db_connector: BaseDatabaseConnector):
        self.db = db_connector

    def get_stock_level(
        self, warehouse_id: int, district_id: int, threshold: int
    ) -> Dict[str, Any]:
        """Execute TPC-C Stock Level transaction"""
        try:
            return self.db.get_stock_level(warehouse_id, district_id, threshold)
        except Exception as e:
            logger.error(f"Stock level service error: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_inventory(
        self,
        warehouse_id: Optional[int] = None,
        low_stock_threshold: int = 10,
        item_search: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get inventory data with optional filters"""
        try:
            return self.db.get_inventory(
                warehouse_id, low_stock_threshold, item_search, limit
            )
        except Exception as e:
            logger.error(f"Get inventory service error: {str(e)}")
            return []

    def get_inventory_paginated(
        self,
        warehouse_id: Optional[int] = None,
        low_stock_threshold: int = 10,
        item_search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """Get inventory data with pagination"""
        try:
            
            # Base query
            query = """
                SELECT 
                    s.s_i_id,
                    s.s_w_id,
                    s.s_quantity,
                    s.s_ytd,
                    s.s_order_cnt,
                    s.s_remote_cnt,
                    i.i_name,
                    i.i_price,
                    i.i_data,
                    w.w_name
                FROM stock s
                JOIN item i ON i.i_id = s.s_i_id
                JOIN warehouse w ON w.w_id = s.s_w_id
                WHERE 1=1
            """
            params = []

            # Apply filters
            if warehouse_id is not None:
                query += " AND s.s_w_id = %s"
                params.append(warehouse_id)
            
            if low_stock_threshold is not None:
                query += " AND s.s_quantity < %s"
                params.append(low_stock_threshold)
                
            if item_search:
                query += " AND (i.i_name ILIKE %s OR i.i_data ILIKE %s)"
                search_term = f"%{item_search}%"
                params.extend([search_term, search_term])

            # Add sorting
            query += " ORDER BY s.s_quantity ASC, i.i_name ASC"
            
            # Add pagination
            query += " LIMIT %s OFFSET %s"
            params.extend([limit, offset])

            # Execute main query
            rows = self.execute_query(query, tuple(params))

            # Count query for pagination
            count_query = """
                SELECT COUNT(*) 
                FROM stock s
                JOIN item i ON i.i_id = s.s_i_id
                WHERE 1=1
            """
            count_params = []

            if warehouse_id is not None:
                count_query += " AND s.s_w_id = %s"
                count_params.append(warehouse_id)
            
            if low_stock_threshold is not None:
                count_query += " AND s.s_quantity < %s"
                count_params.append(low_stock_threshold)
                
            if item_search:
                count_query += " AND (i.i_name ILIKE %s OR i.i_data ILIKE %s)"
                count_params.extend([f"%{item_search}%", f"%{item_search}%"])

            total_count = self.execute_query(count_query, tuple(count_params))[0]['count']

            return {
                "inventory": rows,
                "total_count": total_count,
                "limit": limit,
                "offset": offset,
                "has_next": (offset + limit) < total_count,
                "has_prev": offset > 0
            }

        except Exception as e:
            logger.error(f"Get inventory paginated service error: {str(e)}")
            return {
                "inventory": [],
                "total_count": 0,
                "limit": limit,
                "offset": offset,
                "has_next": False,
                "has_prev": False,
            }

    def get_low_stock_items(
        self, warehouse_id: Optional[int] = None, threshold: int = 10, limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get items with low stock levels"""
        try:
            query = """
                SELECT s.s_i_id, s.s_w_id, s.s_quantity, s.s_ytd, s.s_order_cnt,
                       i.i_name, i.i_price, i.i_data,
                       w.w_name
                FROM stock s
                JOIN item i ON i.i_id = s.s_i_id
                JOIN warehouse w ON w.w_id = s.s_w_id
                WHERE s.s_quantity < %s
            """

            params = [threshold]

            if warehouse_id:
                query += " AND s.s_w_id = %s"
                params.append(warehouse_id)

            query += " ORDER BY s.s_quantity ASC LIMIT %s"
            params.append(limit)

            return self.db.execute_query(query, tuple(params))

        except Exception as e:
            logger.error(f"Get low stock items service error: {str(e)}")
            return []

    def get_item_details(self, item_id: int) -> Dict[str, Any]:
        """Get detailed information about a specific item"""
        try:
            # Get item information
            item_query = """
                SELECT i.*, COUNT(s.s_w_id) as warehouse_count,
                       AVG(s.s_quantity) as avg_stock,
                       MIN(s.s_quantity) as min_stock,
                       MAX(s.s_quantity) as max_stock,
                       SUM(s.s_ytd) as total_ytd,
                       SUM(s.s_order_cnt) as total_orders
                FROM item i
                LEFT JOIN stock s ON s.s_i_id = i.i_id
                WHERE i.i_id = %s
                GROUP BY i.i_id, i.i_im_id, i.i_name, i.i_price, i.i_data
            """

            item_result = self.db.execute_query(item_query, (item_id,))

            if not item_result:
                return {"success": False, "error": "Item not found"}

            item = item_result[0]

            # Get stock by warehouse
            stock_query = """
                SELECT s.s_w_id, s.s_quantity, s.s_ytd, s.s_order_cnt, s.s_remote_cnt,
                       w.w_name, w.w_city, w.w_state
                FROM stock s
                JOIN warehouse w ON w.w_id = s.s_w_id
                WHERE s.s_i_id = %s
                ORDER BY s.s_w_id
            """

            stock_by_warehouse = self.db.execute_query(stock_query, (item_id,))

            return {
                "success": True,
                "item": item,
                "stock_by_warehouse": stock_by_warehouse,
            }

        except Exception as e:
            logger.error(f"Get item details service error: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_inventory_statistics(
        self, warehouse_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """Get inventory statistics"""
        try:
            stats = {}

            # Base query conditions
            where_clause = "WHERE 1=1"
            params = []

            if warehouse_id:
                where_clause += " AND s_w_id = %s"
                params.append(warehouse_id)

            # Total items in stock
            total_query = f"SELECT COUNT(*) as count FROM stock {where_clause}"
            total_result = self.db.execute_query(total_query, tuple(params))
            stats["total_stock_records"] = (
                total_result[0]["count"] if total_result else 0
            )

            # Low stock items (< 10)
            low_stock_query = f"""
                SELECT COUNT(*) as count 
                FROM stock 
                {where_clause} AND s_quantity < 10
            """
            low_stock_result = self.db.execute_query(low_stock_query, tuple(params))
            stats["low_stock_items"] = (
                low_stock_result[0]["count"] if low_stock_result else 0
            )

            # Out of stock items (= 0)
            out_of_stock_query = f"""
                SELECT COUNT(*) as count 
                FROM stock 
                {where_clause} AND s_quantity = 0
            """
            out_of_stock_result = self.db.execute_query(
                out_of_stock_query, tuple(params)
            )
            stats["out_of_stock_items"] = (
                out_of_stock_result[0]["count"] if out_of_stock_result else 0
            )

            # Average stock quantity
            avg_stock_query = f"""
                SELECT AVG(s_quantity) as avg_quantity 
                FROM stock 
                {where_clause}
            """
            avg_stock_result = self.db.execute_query(avg_stock_query, tuple(params))
            stats["avg_stock_quantity"] = (
                float(avg_stock_result[0]["avg_quantity"])
                if avg_stock_result and avg_stock_result[0]["avg_quantity"]
                else 0.0
            )

            # Total inventory value (approximate)
            value_query = f"""
                SELECT SUM(s.s_quantity * i.i_price) as total_value
                FROM stock s
                JOIN item i ON i.i_id = s.s_i_id
                {where_clause}
            """
            value_result = self.db.execute_query(value_query, tuple(params))
            stats["total_inventory_value"] = (
                float(value_result[0]["total_value"])
                if value_result and value_result[0]["total_value"]
                else 0.0
            )

            # Most ordered items
            top_items_query = f"""
                SELECT s.s_i_id, i.i_name, s.s_order_cnt, s.s_quantity
                FROM stock s
                JOIN item i ON i.i_id = s.s_i_id
                {where_clause}
                ORDER BY s.s_order_cnt DESC
                LIMIT 5
            """
            top_items_result = self.db.execute_query(top_items_query, tuple(params))
            stats["top_ordered_items"] = top_items_result

            return stats

        except Exception as e:
            logger.error(f"Get inventory statistics service error: {str(e)}")
            return {}

    def search_items(self, search_term: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search for items by name or data"""
        try:
            query = """
                SELECT DISTINCT i.i_id, i.i_name, i.i_price, i.i_data,
                       COUNT(s.s_w_id) as warehouse_count,
                       AVG(s.s_quantity) as avg_stock,
                       MIN(s.s_quantity) as min_stock
                FROM item i
                LEFT JOIN stock s ON s.s_i_id = i.i_id
                WHERE i.i_name ILIKE %s OR i.i_data ILIKE %s
                GROUP BY i.i_id, i.i_name, i.i_price, i.i_data
                ORDER BY i.i_name
                LIMIT %s
            """

            search_param = f"%{search_term}%"
            return self.db.execute_query(query, (search_param, search_param, limit))

        except Exception as e:
            logger.error(f"Search items service error: {str(e)}")
            return []

    def get_warehouse_inventory_summary(self, warehouse_id: int) -> Dict[str, Any]:
        """Get inventory summary for a specific warehouse"""
        try:
            summary_query = """
                SELECT 
                    COUNT(*) as total_items,
                    SUM(s_quantity) as total_quantity,
                    AVG(s_quantity) as avg_quantity,
                    COUNT(CASE WHEN s_quantity < 10 THEN 1 END) as low_stock_count,
                    COUNT(CASE WHEN s_quantity = 0 THEN 1 END) as out_of_stock_count,
                    SUM(s_ytd) as total_ytd,
                    SUM(s_order_cnt) as total_orders,
                    SUM(s.s_quantity * i.i_price) as total_value
                FROM stock s
                JOIN item i ON i.i_id = s.s_i_id
                WHERE s.s_w_id = %s
            """

            result = self.db.execute_query(summary_query, (warehouse_id,))

            if not result:
                return {"success": False, "error": "Warehouse not found"}

            summary = result[0]

            # Get warehouse info
            warehouse_query = """
                SELECT w_name, w_city, w_state
                FROM warehouse
                WHERE w_id = %s
            """

            warehouse_result = self.db.execute_query(warehouse_query, (warehouse_id,))
            warehouse_info = warehouse_result[0] if warehouse_result else {}

            return {
                "success": True,
                "warehouse_id": warehouse_id,
                "warehouse_info": warehouse_info,
                "summary": summary,
            }

        except Exception as e:
            logger.error(f"Get warehouse inventory summary service error: {str(e)}")
            return {"success": False, "error": str(e)}
