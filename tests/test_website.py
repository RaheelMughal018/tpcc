#!/usr/bin/env python3
"""
TPC-C Website Feature Test Script
Tests all website pages and basic functionality
"""

import sys
from urllib.parse import urljoin

import requests


def test_website_features():
    """Test all website features"""
    print("=" * 60)
    print("TPC-C Website Feature Test")
    print("=" * 60)

    # Base URL for testing
    base_url = "http://localhost:5000"

    # Test pages
    pages_to_test = [
        ("/", "Dashboard"),
        ("/orders", "Orders"),
        ("/inventory", "Inventory"),
        ("/payments", "Payments"),
        ("/test/acid", "ACID Tests"),
        ("/api/health", "Health Check API"),
    ]

    # API endpoints to test
    api_endpoints = [
        ("/api/health", "GET", "Health Check"),
    ]

    print("🌐 Testing Website Pages...")
    print("-" * 40)

    for path, name in pages_to_test:
        try:
            url = urljoin(base_url, path)
            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                print(f"✅ {name}: OK (Status: {response.status_code})")

                # Check for specific content
                if path == "/":
                    if "TPC-C Testing" in response.text:
                        print("   ✓ Dashboard content verified")
                    else:
                        print("   ⚠️  Dashboard content not found")

                elif path == "/api/health":
                    try:
                        health_data = response.json()
                        if "status" in health_data:
                            print(
                                f"   ✓ Health check data: {health_data.get('status', 'unknown')}"
                            )
                        else:
                            print("   ⚠️  Health check format unexpected")
                    except:
                        print("   ⚠️  Health check not returning JSON")

            else:
                print(f"❌ {name}: Failed (Status: {response.status_code})")

        except requests.exceptions.ConnectionError:
            print(f"❌ {name}: Connection failed - Is the server running?")
        except requests.exceptions.Timeout:
            print(f"❌ {name}: Timeout - Server may be slow")
        except Exception as e:
            print(f"❌ {name}: Error - {str(e)}")

    print("\n🔧 Testing Static Assets...")
    print("-" * 40)

    static_assets = [("/static/css/custom.css", "Custom CSS")]

    for path, name in static_assets:
        try:
            url = urljoin(base_url, path)
            response = requests.get(url, timeout=5)

            if response.status_code == 200:
                print(f"✅ {name}: OK (Size: {len(response.content)} bytes)")
            else:
                print(f"❌ {name}: Failed (Status: {response.status_code})")

        except Exception as e:
            print(f"❌ {name}: Error - {str(e)}")

    print("\n📊 Testing Error Pages...")
    print("-" * 40)

    error_pages = [
        ("/nonexistent-page", "404 Error Page"),
    ]

    for path, name in error_pages:
        try:
            url = urljoin(base_url, path)
            response = requests.get(url, timeout=5)

            if response.status_code == 404:
                if "404" in response.text and "Page Not Found" in response.text:
                    print(f"✅ {name}: OK (Custom 404 page working)")
                else:
                    print(f"⚠️  {name}: Default 404 page (custom page not working)")
            else:
                print(f"❌ {name}: Unexpected status {response.status_code}")

        except Exception as e:
            print(f"❌ {name}: Error - {str(e)}")

    print("\n🎨 Testing UI Components...")
    print("-" * 40)

    # Test dashboard for key UI components
    try:
        response = requests.get(base_url, timeout=10)
        if response.status_code == 200:
            content = response.text

            # Check for Bootstrap
            if "bootstrap" in content.lower():
                print("✅ Bootstrap CSS: Loaded")
            else:
                print("❌ Bootstrap CSS: Not found")

            # Check for Bootstrap Icons
            if "bootstrap-icons" in content.lower():
                print("✅ Bootstrap Icons: Loaded")
            else:
                print("❌ Bootstrap Icons: Not found")

            # Check for custom CSS
            if "custom.css" in content:
                print("✅ Custom CSS: Linked")
            else:
                print("❌ Custom CSS: Not linked")

            # Check for navigation
            if (
                "Dashboard" in content
                and "Orders" in content
                and "Inventory" in content
            ):
                print("✅ Navigation: Complete")
            else:
                print("❌ Navigation: Incomplete")

            # Check for provider badge
            if "Google Spanner" in content or "provider-badge" in content:
                print("✅ Provider Badge: Present")
            else:
                print("⚠️  Provider Badge: Not visible")

        else:
            print("❌ UI Components: Cannot test (dashboard not accessible)")

    except Exception as e:
        print(f"❌ UI Components: Error - {str(e)}")

    print("\n" + "=" * 60)
    print("✅ Website feature test completed!")
    print("=" * 60)

    print("\n📋 Test Summary:")
    print("- All major pages should be accessible")
    print("- Navigation should work between pages")
    print("- Custom styling should be applied")
    print("- Error pages should display properly")
    print("- API endpoints should respond correctly")

    print("\n🚀 Next Steps:")
    print("1. Start the Flask application: python app.py")
    print("2. Open browser to: http://localhost:5000")
    print("3. Test TPC-C transactions through the web interface")
    print("4. Verify Google Spanner connectivity")
    print("5. Run ACID compliance tests")


def check_server_running():
    """Check if the Flask server is running"""
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False


if __name__ == "__main__":
    print("Testing TPC-C website features...")

    if not check_server_running():
        print("\n⚠️  Flask server is not running!")
        print("Please start the server first:")
        print("  cd tpcc-webapp")
        print("  python app.py")
        print("\nThen run this test again.")
        sys.exit(1)

    test_website_features()
