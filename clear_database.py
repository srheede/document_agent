import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_SERVICE_KEY"))

def clear_database():
    try:
        # Delete rows where "url" is not empty (i.e., all rows) and return deleted rows
        result = supabase.table("site_pages").delete().neq("url", "").execute()
        deleted_titles = [row.get("title") for row in result.data] if result.data else []
        print("Deleted rows with titles:", deleted_titles)
    except Exception as e:
        print("Error clearing database:", e)

if __name__ == "__main__":
    clear_database()
