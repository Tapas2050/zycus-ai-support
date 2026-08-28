from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Ticket(BaseModel):
    model_config = ConfigDict(extra="allow")

    ticket_id: str
    account_id: str
    company: str
    subject: str
    body: str
    product: str
    product_area: str
    category: str
    urgency: str
    status: str
    plan_tier: str
    assigned_agent: str
    created_at: datetime
    updated_at: datetime
    tags: list[str]
    channel: str
    satisfaction_score: int | None = None


class PrimaryContact(BaseModel):
    name: str
    title: str


class Account(BaseModel):
    model_config = ConfigDict(extra="allow")

    account_id: str
    company: str
    tam: str
    plan_tier: str
    arr_usd: int
    seats_licensed: int
    seats_active: int
    products: list[str]
    health_status: str
    usage_trend: str
    open_tickets: int
    p1_tickets_last_30d: int
    customer_since: str
    renewal_date: str
    last_qbr_date: str
    primary_contact: PrimaryContact
    escalation_notes: list[str]
    nps_score: int | None = None
    last_login_days_ago: int
    integrations_active: list[str]
    region: str
    industry: str


class TicketInput(BaseModel):
    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)


class KBChunk(BaseModel):
    chunk_id: str
    source_file: str
    heading: str | None
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
