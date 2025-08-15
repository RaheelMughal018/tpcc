# TPC-C Database Integration

## Your Task

Implement a database connector to make this TPC-C web application display real data from your assigned database provider: **Amazon Aurora DSQL**.

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure Database Connection**
   ```bash
   cp .env.example .env
   # Edit .env with your database credentials
   ```

3. **Implement Database Connector**
   - Open `database/aurora_connector.py`
   - Follow the TODO comments to implement connection logic
   - Test your implementation

4. **Run the Application**
   ```bash
   python app.py
   ```

5. **Verify Success**
   - Visit http://localhost:5000
   - Check that all pages display real data
   - Ensure no error messages appear

## Success Criteria

✅ Connection test passes  
✅ Dashboard shows real data  
✅ Orders page displays recent orders  
✅ Inventory page shows stock levels  
✅ Payments page shows transaction history  
✅ No error messages in browser or terminal  

## Multi-Region Setup (After Local Success)

Once your local implementation works, prepare for cloud deployment:

### 1. Add Region Tracking Column
```sql
ALTER TABLE orders ADD COLUMN IF NOT EXISTS region_created VARCHAR(50);
```

### 2. Why This Is Needed
- Enables multi-region testing in cloud deployment
- Tracks which region created each order
- Required for distributed database scenarios

### 3. When to Apply
- After local testing passes ✅
- Before cloud deployment
- Run on your database using your preferred SQL client

## Files to Implement

Implement the database connector for your assigned provider:

- `database/aurora_connector.py` - Amazon Aurora DSQL

## Technical Overview

See `materials/shared/webapp-overview.md` for technical details about the application architecture.

## Need Help?

- Check the TODO comments in your connector file
- Review the base connector class in `database/base_connector.py`
- Test your connection with the `/api/health` endpoint

Good luck! 🚀
