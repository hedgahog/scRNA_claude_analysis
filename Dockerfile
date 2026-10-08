# Use an official Python runtime as a parent image
# FROM python:3.8-slim
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Install HDF5 and C build tools needed by Scanpy/AnnData
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade installer tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Leverage Docker cache for dependencies
# Copy the current directory contents into the container at /usr/src/app
COPY requirements.txt .
# Install any needed packages specified in requirements.txt 
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container (Optional, only for web apps)
# EXPOSE 80

RUN useradd -m app && chown app:app /app
COPY --chown=app:app scRNAseq_llm_analysis_step1.py .
USER app

# Copy source code
# COPY . .

# Define environment variable (optional)
# ENV NAME World

# # Setup an app user so the container doesn't run as the root user
# # build time instructions for linux add user
# RUN useradd -m app
# # root user to app user
# USER app

# Run app.py when the container launches
CMD ["python", "scRNAseq_llm_analysis_step1.py"]
