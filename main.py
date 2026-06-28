import json
import os
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate


load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="DarkAtlas Asset Management API",
    description="Attack Surface Monitoring — Asset Management Module",
    version="1.0.0",
)


SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


VALID_TYPES = {"domain", "subdomain", "ip_address", "service", "certificate", "technology"}
VALID_STATUSES = {"active", "stale", "archived"}
VALID_SOURCES = {"import", "scan", "manual"}


class Asset(Base):
    __tablename__ = "assets"

    id = Column(String, primary_key=True, index=True)
    type = Column(String)
    value = Column(String)
    status = Column(String)
    first_seen = Column(String)
    last_seen = Column(String)
    source = Column(String)
    tags = Column(String)
    asset_metadata = Column(String)
    risk_score = Column(Integer, nullable=True)
    environment = Column(String, nullable=True)
    criticality = Column(String, nullable=True)
    ai_summary = Column(String, nullable=True)


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class AssetCreate(BaseModel):
    id: str = Field(description="Unique stable identifier, e.g. 'a1' or a UUID")
    type: str = Field(description="One of: domain, subdomain, ip_address, service, certificate, technology")
    value: str = Field(description="Canonical value, e.g. api.example.com, 203.0.113.10, 443/tcp")
    status: str = Field(description="One of: active, stale, archived")
    first_seen: datetime = Field(description="When the asset was first recorded")
    last_seen: datetime = Field(description="When the asset was last sighted")
    source: str = Field(description="One of: import, scan, manual")
    tags: List[str] = Field(default=[], description="Free-form labels")
    asset_metadata: Dict[str, Any] = Field(default={}, description="Type-specific fields, e.g. cert issuer/expiry, service banner")


class AssetQueryFilters(BaseModel):
    type: Optional[str] = Field(None, description="Asset type, e.g. 'certificate', 'subdomain'")
    status: Optional[str] = Field(None, description="Asset status, e.g. 'active', 'stale'")
    search_term: Optional[str] = Field(None, description="Value or tag to search for, e.g. 'example.com', 'prod'")
    environment: Optional[str] = Field(None, description="Environment classification, e.g. 'production', 'staging'")


class AssetEnrichment(BaseModel):
    risk_score: int = Field(description="Risk score 1-10. 10 = highest risk (e.g. expired cert on prod), 1 = lowest risk.")
    environment: str = Field(description="Classify as 'production', 'staging', or 'dev' based on tags, value, and metadata.")
    criticality: str = Field(description="One of: 'critical', 'high', 'medium', 'low'.")
    ai_summary: str = Field(description="One-sentence risk summary for a security analyst.")
    enriched_metadata: Dict[str, str] = Field(description="2 net-new metadata fields inferred from the asset, e.g. {'exposure': 'public', 'cert_health': 'expired'}.")


class RiskReport(BaseModel):
    executive_summary: str = Field(description="2-3 sentence summary of the overall attack surface posture.")
    critical_findings: List[str] = Field(description="List of specific high-risk findings, e.g. expired certs, exposed sensitive services, EOL technologies.")
    assets_by_risk: Dict[str, List[str]] = Field(description="Assets grouped by risk level: {'critical': [...], 'high': [...], 'medium': [...], 'low': [...]}")
    recommended_actions: List[str] = Field(description="Prioritized list of remediation actions for the security team.")


# ---------------------------------------------------------------------------
# LLM & chains
# ---------------------------------------------------------------------------
llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

enrich_prompt = PromptTemplate.from_template("""
You are a senior attack surface security analyst working on the DarkAtlas ASM platform.

Analyze the following discovered asset and provide a security enrichment:

Asset ID: {id}
Type: {type}
Value: {value}
Status: {status}
Source: {source}
Tags: {tags}
Metadata: {asset_metadata}

Rules:
- risk_score: 1-10 where 10 = critical risk (expired cert, exposed admin service, EOL tech on prod)
- environment: infer from tags/value ('prod'/'production' → production, 'staging'/'stg' → staging, else dev)
- criticality: critical (score 9-10), high (7-8), medium (4-6), low (1-3)
- ai_summary: one sentence a CISO would read in a briefing
- enriched_metadata: 2 new fields that add security context not already in metadata

IMPORTANT: Only use information from the asset record above. Do not invent details.

{error_feedback}
""")

report_prompt = PromptTemplate.from_template("""
You are a senior security analyst preparing an attack surface report for a CISO.

Here is the current asset inventory:

{inventory}

Analyze the full inventory and produce a structured risk report.
Focus on: expired/expiring certificates, exposed sensitive services, end-of-life technologies,
stale assets still showing as active, and high-value targets.

IMPORTANT: Only reference assets that appear in the inventory above. Do not invent assets.

{error_feedback}
""")

query_prompt = PromptTemplate.from_template("""
You are a search assistant for an Attack Surface Monitoring platform.
Translate this natural language query into structured database filters:

Query: "{request}"

Asset types available: domain, subdomain, ip_address, service, certificate, technology
Asset statuses available: active, stale, archived
Environments available: production, staging, dev

Leave any filter null if not mentioned in the query.

{error_feedback}
""")

enrich_chain = enrich_prompt | llm.with_structured_output(AssetEnrichment)
report_chain = report_prompt | llm.with_structured_output(RiskReport)
nl_query_chain = query_prompt | llm.with_structured_output(AssetQueryFilters)


def run_feedback_loop(chain, input_data: dict, max_retries: int = 3):
    input_data["error_feedback"] = ""
    for attempt in range(1, max_retries + 1):
        try:
            return chain.invoke(input_data)
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"AI failed after {max_retries} attempts: {e}")
                raise HTTPException(status_code=500, detail="AI generation failed after max retries.")
            logger.warning(f"Attempt {attempt} failed. Retrying with error feedback...")
            input_data["error_feedback"] = (
                f"\n--- SYSTEM WARNING ---\n"
                f"Previous output failed validation: {e}\n"
                f"Return strictly formatted JSON only."
            )


def asset_to_dict(asset: Asset) -> dict:
    return {
        "id": asset.id,
        "type": asset.type,
        "value": asset.value,
        "status": asset.status,
        "first_seen": asset.first_seen,
        "last_seen": asset.last_seen,
        "source": asset.source,
        "tags": json.loads(asset.tags) if asset.tags else [],
        "metadata": json.loads(asset.asset_metadata) if asset.asset_metadata else {},
        "risk_score": asset.risk_score,
        "environment": asset.environment,
        "criticality": asset.criticality,
        "ai_summary": asset.ai_summary,
    }


def validate_asset(item: AssetCreate):
    if item.type not in VALID_TYPES:
        raise HTTPException(status_code=422, detail=f"Invalid type '{item.type}'. Must be one of: {VALID_TYPES}")
    if item.status not in VALID_STATUSES:
        raise HTTPException(status_code=422, detail=f"Invalid status '{item.status}'. Must be one of: {VALID_STATUSES}")
    if item.source not in VALID_SOURCES:
        raise HTTPException(status_code=422, detail=f"Invalid source '{item.source}'. Must be one of: {VALID_SOURCES}")

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/assets/import", summary="Bulk import assets (idempotent)")
def bulk_import(assets: List[AssetCreate], db: Session = Depends(get_db)):
    """
    Idempotent bulk import. Re-importing an existing asset updates last_seen,
    status, tags, and metadata — it does not create a duplicate.
    Malformed records are skipped gracefully and reported in the response.
    """
    added, updated, skipped = 0, 0, []

    for item in assets:
        try:
            validate_asset(item)
            existing = db.query(Asset).filter(Asset.id == item.id).first()

            if existing:
                existing.last_seen = item.last_seen.isoformat()
                existing.status = item.status
                existing.tags = json.dumps(item.tags)
                existing_meta = json.loads(existing.asset_metadata) if existing.asset_metadata else {}
                existing_meta.update(item.asset_metadata)
                existing.asset_metadata = json.dumps(existing_meta)
                updated += 1
            else:
                db.add(Asset(
                    id=item.id,
                    type=item.type,
                    value=item.value,
                    status=item.status,
                    first_seen=item.first_seen.isoformat(),
                    last_seen=item.last_seen.isoformat(),
                    source=item.source,
                    tags=json.dumps(item.tags),
                    asset_metadata=json.dumps(item.asset_metadata),
                    risk_score=None,
                    environment=None,
                    criticality=None,
                    ai_summary=None,
                ))
                added += 1
        except HTTPException as e:
            skipped.append({"id": item.id, "reason": e.detail})
        except Exception as e:
            skipped.append({"id": item.id, "reason": str(e)})

    db.commit()
    return {"added": added, "updated": updated, "skipped": skipped}


@app.post("/assets/enrich/batch", summary="Batch enrich unenriched assets")
def batch_enrich(batch_size: int = 5, db: Session = Depends(get_db)):
    """
    Finds assets with no risk_score (unenriched) and runs AI enrichment on them.
    Commits all changes in a single transaction.
    """
    pending = db.query(Asset).filter(Asset.risk_score.is_(None)).limit(batch_size).all()
    if not pending:
        return {"message": "All assets are already enriched."}

    for asset in pending:
        ai_result = run_feedback_loop(enrich_chain, {
            "id": asset.id,
            "type": asset.type,
            "value": asset.value,
            "status": asset.status,
            "source": asset.source,
            "tags": asset.tags,
            "asset_metadata": asset.asset_metadata,
        })
        asset.risk_score = ai_result.risk_score
        asset.environment = ai_result.environment
        asset.criticality = ai_result.criticality
        asset.ai_summary = ai_result.ai_summary
        current_meta = json.loads(asset.asset_metadata) if asset.asset_metadata else {}
        current_meta.update(ai_result.enriched_metadata)
        asset.asset_metadata = json.dumps(current_meta)

    db.commit()
    return {
        "message": "Batch enrichment complete.",
        "assets_enriched": len(pending),
        "processed_ids": [a.id for a in pending],
    }


@app.post("/assets/enrich/{asset_id}", summary="Enrich a single asset with AI analysis")
def enrich_asset(asset_id: str, db: Session = Depends(get_db)):
    """
    Runs the AI enrichment pipeline on a single asset and writes results back to the DB.
    """
    asset = db.query(Asset).filter(Asset.id == asset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset not found")

    ai_result = run_feedback_loop(enrich_chain, {
        "id": asset.id,
        "type": asset.type,
        "value": asset.value,
        "status": asset.status,
        "source": asset.source,
        "tags": asset.tags,
        "asset_metadata": asset.asset_metadata,
    })

    # Write enrichment back to DB
    asset.risk_score = ai_result.risk_score
    asset.environment = ai_result.environment
    asset.criticality = ai_result.criticality
    asset.ai_summary = ai_result.ai_summary
    current_meta = json.loads(asset.asset_metadata) if asset.asset_metadata else {}
    current_meta.update(ai_result.enriched_metadata)
    asset.asset_metadata = json.dumps(current_meta)
    db.commit()

    return {
        "asset_id": asset.id,
        "value": asset.value,
        "enrichment": ai_result.model_dump(),
    }


@app.get("/assets/report", summary="Generate AI risk report over the full inventory")
def generate_report(db: Session = Depends(get_db)):
    """
    Generates a structured executive risk report grounded in the actual asset inventory.
    The LLM is explicitly prohibited from referencing assets not in the DB.
    """
    assets = db.query(Asset).all()
    if not assets:
        raise HTTPException(status_code=404, detail="No assets found.")

    
    inventory = "\n".join(
        f"ID: {a.id} | Type: {a.type} | Value: {a.value} | Status: {a.status} "
        f"| Risk: {a.risk_score} | Criticality: {a.criticality} "
        f"| Environment: {a.environment} | Tags: {a.tags} | Metadata: {a.asset_metadata}"
        for a in assets
    )

    ai_report = run_feedback_loop(report_chain, {"inventory": inventory})
    return {
        "report_type": "Attack Surface Risk Report",
        "asset_count": len(assets),
        "report": ai_report.model_dump(),
    }


@app.get("/assets/search", summary="Natural language asset search")
def nl_search(q: str, db: Session = Depends(get_db)):
    """
    Translates a plain-English query into structured DB filters and returns matching assets.
    Example: 'show me all expired certificates on production subdomains'
    """
    filters = run_feedback_loop(nl_query_chain, {"request": q})

    query = db.query(Asset)
    if filters.type:
        query = query.filter(Asset.type.ilike(f"%{filters.type}%"))
    if filters.status:
        query = query.filter(Asset.status.ilike(f"%{filters.status}%"))
    if filters.environment:
        query = query.filter(Asset.environment.ilike(f"%{filters.environment}%"))
    if filters.search_term:
        query = query.filter(
            Asset.value.ilike(f"%{filters.search_term}%") |
            Asset.tags.ilike(f"%{filters.search_term}%")
        )

    results = query.all()
    return {
        "user_query": q,
        "ai_interpreted_filters": filters.model_dump(),
        "total_results": len(results),
        "results": [asset_to_dict(a) for a in results],
    }