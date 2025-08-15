# TPC-C Webapp Technical Overview

## Architecture

This Flask web application displays TPC-C retail data using a direct connector pattern:

```
Flask App (app.py)
    ↓
Analytics Service (services/analytics_service.py)
    ↓
Your Database Connector (database/[provider]_connector.py)
    ↓
Your Database
```

## Database Schema

The TPC-C schema includes these main tables:
- `warehouse` - Warehouse locations
- `customer` - Customer information  
- `orders` - Order records
- `order_line` - Order line items
- `item` - Product catalog
- `stock` - Inventory levels

## Your Implementation

Implement these methods in your connector:

1. **`__init__()`** - Initialize database connection
2. **`test_connection()`** - Test database connectivity
3. **`execute_query()`** - Execute SQL queries
4. **`close_connection()`** - Clean up connections

## Environment Variables

Configure your database connection using environment variables in `.env`:

- Add provider-specific connection settings
- See `.env.example` for templates

## Testing

- Run `python app.py` to start the webapp
- Visit http://localhost:5000 to test
- Check `/api/health` endpoint for connection status
- Verify all pages display real data

The webapp will automatically use your connector implementation.
