# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the local requirements.txt file to the container at /app
COPY pyproject.toml poetry.lock /app/
# Install OpenCV dependencies
RUN apt-get update && apt-get install -y libopencv-dev

# Install Poetry
RUN pip install poetry \
    && poetry config virtualenvs.create false

# Install dependencies from pyproject.toml
RUN poetry install --no-interaction --no-ansi

# Copy the local code to the container at /app
COPY . /app

# Expose port 8501 for Streamlit
EXPOSE 8501

# Run Streamlit app when the container launches
CMD ["streamlit", "run", "src/st_ui.py"]
