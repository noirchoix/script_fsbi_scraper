import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List
import traceback

# External Libraries
import pandas as pd
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlmodel import Session, SQLModel, create_engine, select

# Internal Imports (Ensure functions.py is in the same directory)
from server.functions import build_fsbi_dataframes 
from server.database import (
    Job, 
    FlavorDBBatchTemplate, 
    MoleculesBatchTemplate, 
    create_db_and_tables, 
    engine
)

# --- Configuration & Dependencies ---

# Dependency to yield a database session
def get_session():
    with Session(engine) as session:
        yield session

def estimate_scrape_time(max_compounds: int, delay: float) -> float:
    """Estimates time: max_compounds * delay * ~2.5 (heuristic for safety margin)."""
    return round(max_compounds * delay * 2.5, 2)

# --- Pydantic Schemas for API I/O ---

class ScrapeConfig(BaseModel):
    """Input configuration for a new scrape job."""
    discover_terms: str = Field(..., description="apple, cinnamon, coffee")
    max_compounds: int = Field(500, ge=1, description="Max compounds to process overall.")
    delay: float = Field(0.25, ge=0.01, description="Delay between requests in seconds.")

class JobResponse(BaseModel):
    """Output for displaying job status."""
    id: int
    discover_terms: str
    status: str
    created_at: datetime
    estimated_time: Optional[float]
    total_foods_scraped: int
    last_log_message: str

    class Config:
        # Allows mapping from SQLAlchemy/SQLModel models to Pydantic
        from_attributes = True 

# --- Context Manager for Startup/Shutdown ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize the master Job table
    create_db_and_tables()
    yield
    # Shutdown: (No cleanup needed for simple SQLite example)

app = FastAPI(
    title="FSBI Scraper API", 
    version="1.0", 
    lifespan=lifespan,
    description="Backend for managing and executing FSBI-DB scraping jobs."
)

# --- NEW CORS MIDDLEWARE ---
origins = [
    # Allow the frontend development server (Vite, etc.) running on any port
    "http://localhost:5174", 
    "http://127.0.0.1:5174",
    # You might need to add other ports if Vite tries a different one (e.g., 5175, 5173)
    "http://localhost:5173", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # Allows POST, GET, OPTIONS, etc.
    allow_headers=["*"], # Allows all headers
)

# --- Core Scraper Task Function ---

def run_scraper_task(job_id: Optional[int], config: ScrapeConfig):
    """The function run as a background task to execute the scraping and data insertion."""
    
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job: return

        # 1. Update Job Status & Dynamic Table Setup
        job.status = "RUNNING"
        job.started_at = datetime.utcnow()
        session.add(job)
        session.commit()
        session.refresh(job)
        
        flavor_table_name = f"flavordb_batch_{job_id}"
        molecules_table_name = f"molecules_batch_{job_id}"
        
        # Create the batch-specific tables dynamically
        # NOTE: This creates a unique table class for this job
        # CRITICAL: Wrap dynamic creation in a separate try/except
        try:
            FlavorDBTable = type(flavor_table_name, (FlavorDBBatchTemplate, SQLModel), {"__tablename__": flavor_table_name})
            MoleculesTable = type(molecules_table_name, (MoleculesBatchTemplate, SQLModel), {"__tablename__": molecules_table_name})
            
            FlavorDBTable.metadata.create_all(engine)
            MoleculesTable.metadata.create_all(engine)
        except Exception as e:
            # If this fails, the table names are likely causing an issue (e.g., invalid characters)
            print("!" * 60)
            print(f"FATAL ERROR: Dynamic Table Creation Failed for Job {job_id}")
            traceback.print_exc()
            print("!" * 60)
            
            # We need a robust way to commit the failure here too
            with Session(engine) as error_session:
                job_to_update = error_session.get(Job, job_id)
                if job_to_update:
                    job_to_update.status = "ERROR"
                    job_to_update.last_log_message = f"Table creation failed: {type(e).__name__}"
                    error_session.add(job_to_update)
                    error_session.commit()
            return # Exit the function immediately

        try:
            # 2. Execute Scraper (using the logic from your original run_fsbi)
            job.last_log_message = "Scraping data from FSBI-DB..."
            session.add(job); session.commit()
            
            queries = [q.strip() for q in (config.discover_terms or "").split(",") if q.strip()] or None
            
            flavor_df, molecules_df = build_fsbi_dataframes(
                compound_urls=None,
                food_urls=None,
                discovery_queries=queries,
                max_compounds=config.max_compounds,
                delay=config.delay,
            )

            # --- 3. Prepare and Insert Molecules ---
            # Data: 'pubchem id', 'common name', 'flavor profile'
            molecules_df = molecules_df.rename(columns={
                "pubchem id": "pubchem_id",
                "common name": "name",
                "flavor profile": "flavor_profile",
            })
            
            valid_molecule_records = molecules_df.to_dict('records')
            # Filter records without a pubchem_id (if any scrape failed)
            valid_molecule_records = [r for r in valid_molecule_records if r.get('pubchem_id')]
            
            with session:
                # 3. Prepare and Insert Molecules
                job.last_log_message = f"Inserting {len(molecules_df)} molecules..."
                session.add(job)
                
                # Filter out any records that would cause Pydantic errors (e.g., missing primary key)
                valid_molecule_records = molecules_df.to_dict('records')
                valid_molecule_records = [r for r in valid_molecule_records if r.get('pubchem_id')]

                # CRITICAL GUARD: Only proceed if we have valid data AND the dynamic class exists.
                if MoleculesTable and valid_molecule_records:
                    
                    molecule_insert_list = []
                    for r in valid_molecule_records:
                        # FIX: Explicitly cast the set/list to a string for the JSON column (as previously corrected)
                        instance = MoleculesTable( 
                            pubchem_id=r['pubchem_id'],
                            name=r.get('name', 'N/A'),
                            flavor_profile=str(r.get('flavor_profile', set())),
                        )
                        molecule_insert_list.append(instance)
                    
                    session.add_all(molecule_insert_list)
                    
                else:
                    # If the DataFrame is empty due to the scrape failure, skip insertion cleanly.
                    job.last_log_message = "Scraper returned empty data; skipping molecule insertion."
                    session.add(job)

                
            # --- 4. Prepare and Insert Foods (Flavors) ---
            # Data: 'entity id', 'alias', 'synonyms', 'scientific name', 'category', 'molecules'
            flavor_df = flavor_df.rename(columns={
                "entity id": "entity_id",
                "alias": "name",
                "scientific name": "scientific_name",
                "category": "category",
                "synonyms": "synonyms",
                "molecules": "molecule_ids",
            })
            
            flavor_records = flavor_df.to_dict('records')
            
            job.last_log_message = f"Inserting {len(flavor_records)} food entities..."
            session.add(job)
                
            # CRITICAL GUARD: Only proceed if we have valid data AND the dynamic class exists.
            if FlavorDBTable and flavor_records:
                
                flavor_insert_list = []
                for r in flavor_records:
                    # FIX: Explicitly cast the set/list to a string for JSON columns (as previously corrected)
                    instance = FlavorDBTable(
                        entity_id=r['entity_id'], 
                        name=r.get('name', 'N/A'), 
                        category=r.get('category', 'N/A'),
                        scientific_name=r.get('scientific_name'),
                        synonyms=str(r.get('synonyms', set())),
                        molecule_ids=str(r.get('molecule_ids', set())),
                    )
                    flavor_insert_list.append(instance)

                session.add_all(flavor_insert_list)
                    
            else:
                job.last_log_message = "Scraper returned empty data; skipping food insertion."
                session.add(job)

                
            # 5. Finalize Job Status
            job.total_foods_scraped = len(flavor_df)
            job.total_molecules_scraped = len(molecules_df)
            job.last_log_message = f"Scraping COMPLETED. Found {len(flavor_df)} foods and {len(molecules_df)} molecules."
            job.status = "COMPLETED"
            job.completed_at = datetime.utcnow()
            session.add(job)
            
        except Exception as e:
            # 1. Log to console immediately
            print("-" * 60)
            print(f"ERROR: Job {job_id} failed during scraping or database insertion.")
            traceback.print_exc()
            print("-" * 60)

            # 2. Update status using a fresh, independent session
            # This is VITAL because the original 'session' is now likely broken/rolled back.
            with Session(engine) as error_session:
                    job_to_update = error_session.get(Job, job_id)
                    
                    if job_to_update:
                        job_to_update.status = "ERROR"
                        job_to_update.completed_at = datetime.utcnow()
                        job_to_update.last_log_message = f"Scrape failed: {type(e).__name__}. Check server logs for full traceback."
                        
                        error_session.add(job_to_update)
                        error_session.commit()
                 
        # IMPORTANT: If your `job` object still exists in the scope, it's tied 
        # to the failed outer session. Do NOT attempt to use `session.rollback()` 
        # or `session.commit()` on the outer session after the error. 
        # The `with Session(engine)` block should handle cleanup naturally.
# --- FastAPI Routes ---

# 1. Start a New Scrape Job
@app.post("/api/jobs/", response_model=JobResponse, status_code=201)
async def create_scrape_job(config: ScrapeConfig, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    """Starts a new scraping job in the background."""
    
    # Create initial Job entry
    job = Job(
        discover_terms=config.discover_terms,
        max_compounds=config.max_compounds,
        delay=config.delay,
        estimated_time=estimate_scrape_time(config.max_compounds, config.delay)
    )
    session.add(job)
    session.commit()
    session.refresh(job)
    
    # Start the background task (passing the Job ID and config)
    background_tasks.add_task(run_scraper_task, job.id, config)
    
    return job

# 2. Get Job Status
@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def get_job_status(job_id: int, session: Session = Depends(get_session)):
    """Retrieves the current status and metadata of a scraping job."""
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")
    return job

# 3. Export to CSV (Lazy Export)
@app.get("/api/jobs/{job_id}/export")
def export_job_to_csv(job_id: int, session: Session = Depends(get_session)):
    """
    Exports the scraped data for a specific job to a single CSV file bundle.
    For simplicity, this example exports a combined view of all data.
    """
    job = session.get(Job, job_id)
    if not job or job.status != "COMPLETED":
        raise HTTPException(status_code=404, detail="Job not found or not completed.")

    # 1. Dynamically identify and load the tables
    flavor_table_name = f"flavordb_batch_{job_id}"
    molecules_table_name = f"molecules_batch_{job_id}"
    
    # We will use pandas read_sql_table for easy export
    try:
        # Load data into pandas DataFrames
        flavors_df = pd.read_sql_table(flavor_table_name, con=engine, index_col='entity_id')
        molecules_df = pd.read_sql_table(molecules_table_name, con=engine, index_col='pubchem_id')
        
        # NOTE: A proper export would perform the merge/join here, but for simplicity, 
        # we'll write them separately and package them, or just write a single master flavor file.
        
        output_filename = f"fsbi_data_job_{job_id}_flavors.csv"
        flavors_df.to_csv(output_filename, index=True)
        
        # 2. Return FileResponse
        return FileResponse(
            path=output_filename, 
            filename=output_filename, 
            media_type='text/csv'
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


# 4. Get All Jobs
@app.get("/api/jobs/", response_model=List[JobResponse])
def get_all_jobs(session: Session = Depends(get_session)):
    """Retrieves a list of all historical scraping jobs."""
    jobs = session.exec(select(Job).order_by(Job.created_at.desc())).all()
    return jobs