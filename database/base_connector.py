"""
Simplified base database connector for UX study
Only includes essential methods that participants need to implement
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BaseDatabaseConnector(ABC):
    """
    Simplified abstract base class for database connectors in UX study

    Participants only need to implement these 3 essential methods:
    1. __init__ - Initialize connection
    2. test_connection - Test if connection works
    3. execute_query - Execute SQL queries
    """

    def __init__(self):
        self.connection = None
        self.provider_name = "Unknown"

    @abstractmethod
    def test_connection(self) -> bool:
        """Test if database connection is working"""
        pass

    @abstractmethod
    def execute_query(
        self, query: str, params: Optional[tuple] = None
    ) -> List[Dict[str, Any]]:
        """Execute a query and return results as list of dictionaries"""
        pass

    def get_provider_name(self) -> str:
        """Get the database provider name"""
        return self.provider_name

    def close_connection(self):
        """Close database connection - optional to implement"""
        pass
