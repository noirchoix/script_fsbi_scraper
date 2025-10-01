# In your Python interpreter:
from server.functions import build_fsbi_dataframes
from server.database import Job # Just for context, we don't need the session here

# Run the failing call:
# Use a delay of 5.0 seconds to eliminate rate-limiting as the immediate cause
try:
    flavor_df, molecules_df = build_fsbi_dataframes(
        compound_urls=None,
        food_urls=None,
        discovery_queries=['apple', 'cinnamon'],
        max_compounds=10, # Use a small number
        delay=5.0, # Use a huge delay
    )
    print("Scraper succeeded!")
    print(f"Foods found: {len(flavor_df)}")
except Exception as e:
    print(f"Scraper FAILED with error: {e}")