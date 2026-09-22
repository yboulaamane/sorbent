from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from sorbent.config import Settings
from sorbent.main import create_app


@pytest.fixture
def settings() -> Settings:
    # One worker and a tiny chunk size so tests exercise the chunking path
    # without paying for a full pool.
    return Settings(worker_processes=1, chunk_size=5, max_library_size=1000)


@pytest.fixture
async def client(settings: Settings):
    app = create_app(settings)
    transport = ASGITransport(app=app)
    # The context manager is what triggers the lifespan - without it,
    # app.state.store never exists and every route 500s on the dependency.
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        async with app.router.lifespan_context(app):
            yield c


@pytest.fixture
def aspirin() -> str:
    return "CC(=O)Oc1ccccc1C(=O)O"


@pytest.fixture
def caffeine() -> str:
    return "Cn1cnc2c1c(=O)n(C)c(=O)n2C"


@pytest.fixture
def small_library() -> list[str]:
    return [
        "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
        "Cn1cnc2c1c(=O)n(C)c(=O)n2C",  # caffeine
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # ibuprofen
        "CN1CCC[C@H]1c1cccnc1",  # nicotine
        "OC(=O)c1ccccc1O",  # salicylic acid
    ]
