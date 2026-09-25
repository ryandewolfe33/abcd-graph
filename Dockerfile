FROM ghcr.io/astral-sh/uv:alpine3.23 AS build

# Upgrade apk package repo
RUN apk update && apk upgrade --no-cache

# Install build tools + curl
RUN apk add --no-cache \
    bash \
    curl \
    libffi-dev \
    build-base \
    linux-headers

# Choose the type of installation (default - just the base package)
ARG INSTALL_TYPE=normal

WORKDIR /build

# Keeps Python from buffering stdout and stderr to avoid situations where
# the application crashes without emitting any logs due to buffering.
ENV PYTHONUNBUFFERED=1

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Omit development dependencies
ENV UV_NO_DEV=1

# Ensure installed tools can be executed out of the box
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# Install the project's dependencies using the lockfile and settings
RUN if [ "$INSTALL_TYPE" = "normal" ]; then \
        --mount=type=cache,target=/root/.cache/uv \
        --mount=type=bind,source=uv.lock,target=uv.lock \
        --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
        uv sync --locked --no-install-project ; \
    else \
        --mount=type=cache,target=/root/.cache/uv \
        --mount=type=bind,source=uv.lock,target=uv.lock \
        --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
        uv sync --locked --no-install-project --extra $INSTALL_TYPE ;\
    fi

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY pyproject.toml README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked

COPY src src


FROM ghcr.io/astral-sh/uv:python3.12-alpine3.23 AS runtime

# Add a non-root user
RUN addgroup -S abcd && adduser -S abcd -G abcd

WORKDIR /home/abcd-graph

# Copy the installed virtual environment from the build stage
COPY --from=build /.venv /.venv

# Add a default shell
SHELL ["/bin/sh", "-c"]

# Set environment to use venv
ENV PATH="/.venv/bin:$PATH"

# Use non-root user
USER abcd

# Default to python REPL
ENTRYPOINT ["uv run python"]
