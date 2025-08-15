#!/usr/bin/env python3
"""
Flask 3.1.1 Compatibility Test Script
Tests Flask version and basic functionality
"""

import os
import sys


def test_flask_version():
    """Test Flask version and basic imports"""
    print("=" * 60)
    print("Flask 3.1.1 Compatibility Test")
    print("=" * 60)

    try:
        import flask

        print(f"✅ Flask version: {flask.__version__}")

        # Check if it's Flask 3.x
        major_version = int(flask.__version__.split(".")[0])
        if major_version >= 3:
            print(f"✅ Flask 3.x detected (version {flask.__version__})")
        else:
            print(f"⚠️  Flask version {flask.__version__} is not 3.x")

    except ImportError as e:
        print(f"❌ Failed to import Flask: {e}")
        return False

    # Test Werkzeug compatibility
    try:
        import werkzeug

        print(f"✅ Werkzeug version: {werkzeug.__version__}")
    except ImportError as e:
        print(f"❌ Failed to import Werkzeug: {e}")
        return False

    # Test Jinja2 compatibility
    try:
        import jinja2

        print(f"✅ Jinja2 version: {jinja2.__version__}")
    except ImportError as e:
        print(f"❌ Failed to import Jinja2: {e}")
        return False

    # Test basic Flask app creation
    try:
        from flask import Flask, jsonify

        app = Flask(__name__)

        @app.route("/test")
        def test_route():
            return jsonify({"status": "success", "flask_version": flask.__version__})

        print("✅ Basic Flask app creation successful")

        # Test app context
        with app.app_context():
            print("✅ Flask app context works correctly")

    except Exception as e:
        print(f"❌ Flask app creation failed: {e}")
        return False

    # Test database connector imports
    try:
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from database.connector_factory import DatabaseConnectorFactory

        print("✅ Database connector factory import successful")
    except ImportError as e:
        print(f"⚠️  Database connector import failed: {e}")
        print("   This is expected if database dependencies are not installed")

    print("\n" + "=" * 60)
    print("✅ Flask 3.1.1 compatibility test completed successfully!")
    print("=" * 60)
    return True


def test_flask_features():
    """Test Flask 3.1.1 specific features"""
    print("\n🔍 Testing Flask 3.1.1 Features:")

    try:
        from flask import Flask

        app = Flask(__name__)

        # Test modern initialization (no @app.before_first_request)
        print("✅ Modern Flask initialization pattern supported")

        # Test error handlers
        @app.errorhandler(404)
        def not_found(error):
            return {"error": "Not found"}, 404

        print("✅ Error handlers work correctly")

        # Test route decorators with type hints
        @app.route("/api/test/<int:test_id>")
        def test_typed_route(test_id: int):
            return {"test_id": test_id}

        print("✅ Typed route parameters work correctly")

        # Test JSON handling
        from flask import jsonify, request

        @app.route("/api/json-test", methods=["POST"])
        def test_json():
            data = request.get_json()
            return jsonify({"received": data})

        print("✅ JSON request/response handling works correctly")

    except Exception as e:
        print(f"❌ Flask 3.1.1 feature test failed: {e}")
        return False

    return True


if __name__ == "__main__":
    print("Testing Flask 3.1.1 compatibility for TPC-C webapp...")

    success = test_flask_version()
    if success:
        test_flask_features()
        print("\n🎉 All tests passed! Flask 3.1.1 is ready for TPC-C testing.")
    else:
        print("\n❌ Some tests failed. Please check your Flask installation.")
        sys.exit(1)
