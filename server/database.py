from typing import Optional, List, Dict, Any 
from sqlmodel import SQLModel, Field, create_engine
from datetime import datetime
from sqlalchemy import Column
from sqlalchemy.dialects.sqlite import JSON 

# --- 1. Master Job Table (UNMODIFIED) ---
class Job(SQLModel, table=True):
    """
    Tracks the status and configuration of a scraping job.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Configuration
    discover_terms: str = Field(index=True)
    max_compounds: int
    delay: float
    
    # Status and Metadata
    status: str = Field(default="PENDING", index=True) # PENDING, RUNNING, COMPLETED, ERROR
    batch_size: int = Field(default=1000)
    
    # Time Tracking
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    estimated_time: Optional[float] = None
    
    # Result Counts
    total_foods_scraped: int = Field(default=0)
    total_molecules_scraped: int = Field(default=0)
    
    # Log/Error
    last_log_message: str = Field(default="Job created.")


# --- 2. Template for Batch-Specific Data Tables (REVISED) ---

class FlavorDBBatchTemplate(SQLModel): 
    """
    Template for the food entity (FlavorDB) table, unique to each batch/job.
    Matches features: entity id,alias,synonyms,scientific name,category,molecules
    """
    entity_id: int = Field(primary_key=True) 
    
    # Mapped from 'alias'
    name: str 
    
    # Mapped from 'category'
    category: str
    
    # Mapped from 'scientific name' (added for completeness)
    scientific_name: Optional[str] = None
    
    # Mapped from 'synonyms'. Use JSON type as it's a set/list of strings.
    synonyms: str = Field(sa_column=Column(JSON)) # Store as JSON/string representation of set
    
    # Mapped from 'molecules' (The pubchem_ids). Use JSON type as it's a set/list of ints.
    molecule_ids: str = Field(sa_column=Column(JSON)) # Store as JSON/string representation of set    
    # Removed raw_data for flavor entities as it's not strictly necessary based on your feedback.


class MoleculesBatchTemplate(SQLModel):
    """
    Template for the molecule table, unique to each batch/job.
    Matches features: pubchem id,common name,flavor profile
    """
    pubchem_id: int = Field(primary_key=True)
    
    # Mapped from 'common name'
    name: str 
    
    # Mapped from 'flavor profile'. Use JSON type as it's a set of strings.
    flavor_profile: str = Field(sa_column=Column(JSON)) # Store as JSON/string representation of set
    
    # Removed iupac_name, smiles, and raw_data as they are not scraped and not needed for storage.
    

# --- Database Setup (Example for SQLite) ---
sqlite_file_name = "database.db"
sqlite_url = f"sqlite:///{sqlite_file_name}"

engine = create_engine(sqlite_url, echo=True)

def create_db_and_tables():
    """Initializes the database and the master Job table."""
    SQLModel.metadata.create_all(engine)