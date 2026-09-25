FROM ghcr.io/astral-sh/uv:trixie-slim AS build

# Install build tools + curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    libffi-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Choose the type of installation (default - just the base package)
ARG INSTALL_TYPE=normal

WORKDIR /build

# Keeps Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_NO_DEV=1
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=build/uv.lock \
    --mount=type=bind,source=pyproject.toml,target=build/pyproject.toml \
    if [ "$INSTALL_TYPE" = "normal" ]; then \
         \
        uv sync --locked --no-install-project ; \
    else \
        uv sync --locked --no-install-project --extra $INSTALL_TYPE ; \
    fi

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

COPY abcd_graph abcd_graph


FROM ghcr.io/astral-sh/uv:python3.12-trixie-slim AS runtime

# Add a non-root user (Debian syntax)
RUN useradd -m -s /bin/bash abcd

WORKDIR /home/abcd-graph

# Copy the installed virtual environment from the build stage
COPY --from=build /build/.venv /.venv

# Add a default shell
SHELL ["/bin/sh", "-c"]

# Set environment to use venv
ENV PATH="/.venv/bin:$PATH"

# Use non-root user
USER abcd

# Default to python REPL
ENTRYPOINT ["python"]
