FROM python:3.12-slim

# Install system essentials, sudo, and git (required by VS Code server)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    sudo \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user matching standard Dev Container conventions
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Upgrade pip globally
RUN pip install --no-cache-dir --upgrade pip
