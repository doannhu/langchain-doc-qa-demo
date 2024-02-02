FROM python:3.11-slim

# Set environment variables (e.g., set Python to run in unbuffered mode)
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

COPY requirements.txt /app/

RUN pip install -r /app/requirements.txt

# Copy application code into the container
COPY . /app/

EXPOSE 8000


CMD ["python", "-m", "chainlit", "run", "document_qa.py", "-h", "--port", "8000"]