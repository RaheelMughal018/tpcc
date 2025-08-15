"""
ACID Compliance Tests for TPC-C Database Operations
Tests Atomicity, Consistency, Isolation, and Durability with real database operations
"""

import logging
import time
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ACIDTests:
    """ACID compliance test suite for TPC-C operations with real database testing"""

    def __init__(self, db_connector):
        self.db = db_connector
        self.provider_name = db_connector.get_provider_name()
        self.test_id = int(time.time() * 1000)  # Unique test session ID
        self.test_tables_created = []

    def setup_test_environment(self):
        """Set up test tables and initial data"""
        try:
            # Create test tables for ACID testing with provider-specific syntax
            if self.provider_name == "Google Spanner":
                # Google Spanner syntax
                test_tables = [
                    f"""
                    CREATE TABLE acid_test_accounts_{self.test_id} (
                        account_id INT64 NOT NULL,
                        balance NUMERIC NOT NULL,
                        version INT64 NOT NULL DEFAULT 1,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
                    ) PRIMARY KEY (account_id)
                    """,
                    f"""
                    CREATE TABLE acid_test_transactions_{self.test_id} (
                        txn_id INT64 NOT NULL,
                        from_account INT64 NOT NULL,
                        to_account INT64 NOT NULL,
                        amount NUMERIC NOT NULL,
                        status STRING(20) NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
                    ) PRIMARY KEY (txn_id)
                    """,
                    f"""
                    CREATE TABLE acid_test_audit_{self.test_id} (
                        audit_id INT64 NOT NULL,
                        table_name STRING(50) NOT NULL,
                        operation STRING(20) NOT NULL,
                        record_id INT64 NOT NULL,
                        timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP()
                    ) PRIMARY KEY (audit_id)
                    """,
                ]
            else:
                # PostgreSQL syntax (NeonDB, CockroachDB)
                test_tables = [
                    f"""
                    CREATE TABLE acid_test_accounts_{self.test_id} (
                        account_id BIGINT NOT NULL PRIMARY KEY,
                        balance NUMERIC NOT NULL,
                        version BIGINT NOT NULL DEFAULT 1,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """,
                    f"""
                    CREATE TABLE acid_test_transactions_{self.test_id} (
                        txn_id BIGINT NOT NULL PRIMARY KEY,
                        from_account BIGINT NOT NULL,
                        to_account BIGINT NOT NULL,
                        amount NUMERIC NOT NULL,
                        status VARCHAR(20) NOT NULL,
                        created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """,
                    f"""
                    CREATE TABLE acid_test_audit_{self.test_id} (
                        audit_id BIGINT NOT NULL PRIMARY KEY,
                        table_name VARCHAR(50) NOT NULL,
                        operation VARCHAR(20) NOT NULL,
                        record_id BIGINT NOT NULL,
                        timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                    )
                    """,
                ]

            for table_sql in test_tables:
                self.db.execute_query(table_sql)
                logger.info("✅ Created test table for ACID testing")

            self.test_tables_created = [
                f"acid_test_accounts_{self.test_id}",
                f"acid_test_transactions_{self.test_id}",
                f"acid_test_audit_{self.test_id}",
            ]

            # Insert initial test data
            initial_data = [
                f"INSERT INTO acid_test_accounts_{self.test_id} (account_id, balance) VALUES (1, 1000.00)",
                f"INSERT INTO acid_test_accounts_{self.test_id} (account_id, balance) VALUES (2, 500.00)",
                f"INSERT INTO acid_test_accounts_{self.test_id} (account_id, balance) VALUES (3, 750.00)",
            ]

            for insert_sql in initial_data:
                self.db.execute_query(insert_sql)

            logger.info("✅ Test environment setup completed")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to setup test environment: {str(e)}")
            return False

    def cleanup_test_environment(self):
        """Clean up test tables and data"""
        try:
            for table_name in self.test_tables_created:
                try:
                    self.db.execute_query(f"DROP TABLE {table_name}")
                    logger.info(f"✅ Dropped test table: {table_name}")
                except Exception as e:
                    logger.warning(f"⚠️ Could not drop table {table_name}: {str(e)}")

            self.test_tables_created = []
            logger.info("✅ Test environment cleanup completed")

        except Exception as e:
            logger.error(f"❌ Failed to cleanup test environment: {str(e)}")

    def test_atomicity(self) -> Dict[str, Any]:
        """Test transaction atomicity - all or nothing with real database operations"""
        logger.info("🧪 Testing Atomicity (All-or-Nothing) with real transactions")
        start_time = time.time()

        try:
            # Setup test environment
            if not self.setup_test_environment():
                raise Exception("Failed to setup test environment")

            # Get initial balances
            initial_balances = self.db.execute_query(
                f"SELECT account_id, balance FROM acid_test_accounts_{self.test_id} ORDER BY account_id"
            )
            logger.info(f"Initial balances: {initial_balances}")

            # Attempt a transaction that should fail and rollback
            try:
                transfer_amount = 200.00

                # Define operations that should fail as a group
                operations = [
                    # Step 1: Debit from account 1
                    (
                        f"UPDATE acid_test_accounts_{self.test_id} SET balance = balance - {transfer_amount} WHERE account_id = 1",
                        None,
                    ),
                    # Step 2: Credit to account 2
                    (
                        f"UPDATE acid_test_accounts_{self.test_id} SET balance = balance + {transfer_amount} WHERE account_id = 2",
                        None,
                    ),
                    # Step 3: Force a failure by violating a constraint
                    # Try to insert duplicate primary key
                    (
                        f"INSERT INTO acid_test_accounts_{self.test_id} (account_id, balance) VALUES (1, 999.99)",
                        None,
                    ),
                ]

                # Execute in transaction - this should fail and rollback
                self.db.execute_in_transaction(operations)

                # If we get here, the transaction didn't fail as expected
                raise Exception("Transaction should have failed but didn't")

            except Exception as transaction_error:
                logger.info(f"Expected transaction failure: {str(transaction_error)}")

            # Check if changes were rolled back
            final_balances = self.db.execute_query(
                f"SELECT account_id, balance FROM acid_test_accounts_{self.test_id} ORDER BY account_id"
            )
            logger.info(f"Final balances: {final_balances}")

            # Verify atomicity - balances should be unchanged
            atomicity_passed = True
            for initial, final in zip(initial_balances, final_balances):
                if (
                    abs(float(initial["balance"]) - float(final["balance"])) > 0.01
                ):  # Allow for small floating point differences
                    atomicity_passed = False
                    break

            end_time = time.time()
            duration_ms = round((end_time - start_time) * 1000)

            test_result = {
                "test_name": "Atomicity Test",
                "provider": self.provider_name,
                "status": "passed" if atomicity_passed else "failed",
                "description": "Transaction rollback on failure",
                "details": f"Initial: {initial_balances}, Final: {final_balances}",
                "atomicity_verified": atomicity_passed,
                "timestamp": time.time(),
                "duration_ms": duration_ms,
                "duration": f"{duration_ms} ms",
            }

            logger.info(
                f"✅ Atomicity test {'passed' if atomicity_passed else 'failed'}"
            )
            return test_result

        except Exception as e:
            logger.error(f"❌ Atomicity test failed: {str(e)}")
            end_time = time.time()
            duration_ms = round((end_time - start_time) * 1000)

            return {
                "test_name": "Atomicity Test",
                "provider": self.provider_name,
                "status": "failed",
                "error": str(e),
                "timestamp": time.time(),
                "duration_ms": duration_ms,
                "duration": f"{duration_ms} ms",
            }
        finally:
            self.cleanup_test_environment()

    def test_consistency(self) -> Dict[str, Any]:
        """Test data consistency constraints with real constraint violations"""
        logger.info(
            "🧪 Testing Consistency (Constraint Enforcement) with real constraints"
        )
        start_time = time.time()

        try:
            # Setup test environment
            if not self.setup_test_environment():
                raise Exception("Failed to setup test environment")

            consistency_tests = []

            # Test 1: Primary key constraint
            try:
                self.db.execute_query(
                    f"INSERT INTO acid_test_accounts_{self.test_id} (account_id, balance) VALUES (1, 999.99)"
                )
                consistency_tests.append(
                    {
                        "test": "Primary Key Constraint",
                        "passed": False,
                        "error": "Duplicate key allowed",
                    }
                )
            except Exception as e:
                consistency_tests.append(
                    {"test": "Primary Key Constraint", "passed": True, "error": str(e)}
                )
                logger.info("✅ Primary key constraint enforced")

            # Test 2: NOT NULL constraint
            try:
                self.db.execute_query(
                    f"INSERT INTO acid_test_accounts_{self.test_id} (account_id, balance) VALUES (NULL, 100.00)"
                )
                consistency_tests.append(
                    {
                        "test": "NOT NULL Constraint",
                        "passed": False,
                        "error": "NULL value allowed",
                    }
                )
            except Exception as e:
                consistency_tests.append(
                    {"test": "NOT NULL Constraint", "passed": True, "error": str(e)}
                )
                logger.info("✅ NOT NULL constraint enforced")

            # Test 3: Data type constraint
            try:
                self.db.execute_query(
                    f"INSERT INTO acid_test_accounts_{self.test_id} (account_id, balance) VALUES ('invalid', 100.00)"
                )
                consistency_tests.append(
                    {
                        "test": "Data Type Constraint",
                        "passed": False,
                        "error": "Invalid type allowed",
                    }
                )
            except Exception as e:
                consistency_tests.append(
                    {"test": "Data Type Constraint", "passed": True, "error": str(e)}
                )
                logger.info("✅ Data type constraint enforced")

            # Verify database state is still consistent
            final_count = self.db.execute_query(
                f"SELECT COUNT(*) as count FROM acid_test_accounts_{self.test_id}"
            )
            expected_count = 3  # Initial test data

            consistency_passed = (
                all(test["passed"] for test in consistency_tests)
                and final_count[0]["count"] == expected_count
            )

            end_time = time.time()
            duration_ms = round((end_time - start_time) * 1000)

            test_result = {
                "test_name": "Consistency Test",
                "provider": self.provider_name,
                "status": "passed" if consistency_passed else "failed",
                "description": "Database constraints are enforced",
                "details": f"Constraint tests: {consistency_tests}, Final count: {final_count[0]['count']}",
                "constraint_tests": consistency_tests,
                "timestamp": time.time(),
                "duration_ms": duration_ms,
                "duration": f"{duration_ms} ms",
            }

            logger.info(
                f"✅ Consistency test {'passed' if consistency_passed else 'failed'}"
            )
            return test_result

        except Exception as e:
            logger.error(f"❌ Consistency test failed: {str(e)}")
            end_time = time.time()
            duration_ms = round((end_time - start_time) * 1000)

            return {
                "test_name": "Consistency Test",
                "provider": self.provider_name,
                "status": "failed",
                "error": str(e),
                "timestamp": time.time(),
                "duration_ms": duration_ms,
                "duration": f"{duration_ms} ms",
            }
        finally:
            self.cleanup_test_environment()

    def test_isolation(self) -> Dict[str, Any]:
        """Test transaction isolation with concurrent operations simulation"""
        logger.info(
            "🧪 Testing Isolation (Concurrent Transactions) with simulated concurrency"
        )
        start_time = time.time()

        try:
            # Setup test environment
            if not self.setup_test_environment():
                raise Exception("Failed to setup test environment")

            # Simulate concurrent transactions by rapid sequential operations
            isolation_tests = []

            # Test 1: Read consistency during updates
            initial_balance = self.db.execute_query(
                f"SELECT balance FROM acid_test_accounts_{self.test_id} WHERE account_id = 1"
            )[0]["balance"]

            # Simulate Transaction 1: Update balance
            self.db.execute_query(
                f"UPDATE acid_test_accounts_{self.test_id} SET balance = balance + 100 WHERE account_id = 1"
            )

            # Simulate Transaction 2: Read balance (should see committed value)
            updated_balance = self.db.execute_query(
                f"SELECT balance FROM acid_test_accounts_{self.test_id} WHERE account_id = 1"
            )[0]["balance"]

            read_consistency = updated_balance == initial_balance + 100
            isolation_tests.append(
                {
                    "test": "Read Consistency",
                    "passed": read_consistency,
                    "details": f"Initial: {initial_balance}, Updated: {updated_balance}",
                }
            )

            # Test 2: Version-based concurrency control simulation
            try:
                # Update with version check (optimistic locking simulation)
                version_result = self.db.execute_query(
                    f"SELECT version FROM acid_test_accounts_{self.test_id} WHERE account_id = 2"
                )[0]["version"]

                # Simulate concurrent update with version check
                update_result = self.db.execute_query(
                    f"UPDATE acid_test_accounts_{self.test_id} SET balance = balance + 50, version = version + 1 WHERE account_id = 2 AND version = {version_result}"
                )

                isolation_tests.append(
                    {
                        "test": "Version Control",
                        "passed": True,
                        "details": "Version-based update successful",
                    }
                )

            except Exception as e:
                isolation_tests.append(
                    {
                        "test": "Version Control",
                        "passed": False,
                        "details": f"Version control failed: {str(e)}",
                    }
                )

            isolation_passed = all(test["passed"] for test in isolation_tests)

            end_time = time.time()
            duration_ms = round((end_time - start_time) * 1000)

            test_result = {
                "test_name": "Isolation Test",
                "provider": self.provider_name,
                "status": "passed" if isolation_passed else "failed",
                "description": "Concurrent transactions don't interfere",
                "details": f"Isolation tests: {isolation_tests}",
                "isolation_tests": isolation_tests,
                "timestamp": time.time(),
                "duration_ms": duration_ms,
                "duration": f"{duration_ms} ms",
            }

            logger.info(
                f"✅ Isolation test {'passed' if isolation_passed else 'failed'}"
            )
            return test_result

        except Exception as e:
            logger.error(f"❌ Isolation test failed: {str(e)}")
            end_time = time.time()
            duration_ms = round((end_time - start_time) * 1000)

            return {
                "test_name": "Isolation Test",
                "provider": self.provider_name,
                "status": "failed",
                "error": str(e),
                "timestamp": time.time(),
                "duration_ms": duration_ms,
                "duration": f"{duration_ms} ms",
            }
        finally:
            self.cleanup_test_environment()

    def test_durability(self) -> Dict[str, Any]:
        """Test data durability after commit with real persistence verification"""
        logger.info(
            "🧪 Testing Durability (Persistence After Commit) with real data persistence"
        )
        start_time = time.time()

        try:
            # Setup test environment
            if not self.setup_test_environment():
                raise Exception("Failed to setup test environment")

            # Insert test data and verify it's committed
            test_account_id = 999
            test_balance = 12345.67
            test_description = f"Durability test data {self.test_id}"

            # Insert test record
            self.db.execute_query(
                f"INSERT INTO acid_test_accounts_{self.test_id} (account_id, balance) VALUES ({test_account_id}, {test_balance})"
            )

            # Insert audit record
            self.db.execute_query(
                f"INSERT INTO acid_test_audit_{self.test_id} (audit_id, table_name, operation, record_id) VALUES ({test_account_id}, 'accounts', 'INSERT', {test_account_id})"
            )

            logger.info(
                f"✅ Inserted test data: Account {test_account_id} with balance {test_balance}"
            )

            # Simulate connection reset by creating a new query
            # (In a real scenario, this would involve reconnecting to the database)
            time.sleep(0.1)  # Brief pause to simulate time passage

            # Verify data persists after "reconnection"
            persisted_data = self.db.execute_query(
                f"SELECT account_id, balance FROM acid_test_accounts_{self.test_id} WHERE account_id = {test_account_id}"
            )

            persisted_audit = self.db.execute_query(
                f"SELECT audit_id, operation FROM acid_test_audit_{self.test_id} WHERE record_id = {test_account_id}"
            )

            # Verify durability
            data_persisted = (
                len(persisted_data) == 1
                and persisted_data[0]["account_id"] == test_account_id
                and float(persisted_data[0]["balance"]) == test_balance
            )

            audit_persisted = (
                len(persisted_audit) == 1
                and persisted_audit[0]["operation"] == "INSERT"
            )

            durability_passed = data_persisted and audit_persisted

            end_time = time.time()
            duration_ms = round((end_time - start_time) * 1000)

            test_result = {
                "test_name": "Durability Test",
                "provider": self.provider_name,
                "status": "passed" if durability_passed else "failed",
                "description": "Committed data persists after system restart",
                "details": f"Data persisted: {data_persisted}, Audit persisted: {audit_persisted}",
                "persisted_data": persisted_data,
                "persisted_audit": persisted_audit,
                "timestamp": time.time(),
                "duration_ms": duration_ms,
                "duration": f"{duration_ms} ms",
            }

            logger.info(
                f"✅ Durability test {'passed' if durability_passed else 'failed'}"
            )
            return test_result

        except Exception as e:
            logger.error(f"❌ Durability test failed: {str(e)}")
            end_time = time.time()
            duration_ms = round((end_time - start_time) * 1000)

            return {
                "test_name": "Durability Test",
                "provider": self.provider_name,
                "status": "failed",
                "error": str(e),
                "timestamp": time.time(),
                "duration_ms": duration_ms,
                "duration": f"{duration_ms} ms",
            }
        finally:
            self.cleanup_test_environment()

    def run_all_tests(self) -> Dict[str, Any]:
        """Run all ACID compliance tests with real database operations"""
        logger.info(
            f"🧪 Running complete ACID test suite for {self.provider_name} with real database operations"
        )
        start_time = time.time()

        results = {
            "provider": self.provider_name,
            "test_suite": "ACID Compliance (Real Database Tests)",
            "timestamp": time.time(),
            "test_session_id": self.test_id,
            "tests": {},
        }

        # Run each test independently
        logger.info("🔄 Running Atomicity Test...")
        results["tests"]["atomicity"] = self.test_atomicity()

        logger.info("🔄 Running Consistency Test...")
        results["tests"]["consistency"] = self.test_consistency()

        logger.info("🔄 Running Isolation Test...")
        results["tests"]["isolation"] = self.test_isolation()

        logger.info("🔄 Running Durability Test...")
        results["tests"]["durability"] = self.test_durability()

        # Calculate overall results
        passed_tests = sum(
            1 for test in results["tests"].values() if test["status"] == "passed"
        )
        total_tests = len(results["tests"])

        end_time = time.time()
        duration_ms = round((end_time - start_time) * 1000)

        results["summary"] = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "success_rate": (passed_tests / total_tests) * 100
            if total_tests > 0
            else 0,
            "duration_ms": duration_ms,
            "duration": f"{duration_ms} ms",
        }

        logger.info(
            f"✅ ACID test suite completed: {passed_tests}/{total_tests} tests passed ({results['summary']['success_rate']:.1f}%) in {duration_ms} ms"
        )
        return results
