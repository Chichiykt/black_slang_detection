FROM pytorch/pytorch:2.12.0-cuda12.6-cudnn9-runtime
WORKDIR /app

RUN rm /usr/lib/python3.12/EXTERNALLY-MANAGED

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && apt-get install -y curl \
    && curl -fsSL https://ollama.com/install.sh | sh

COPY . .
COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 3306 8080

CMD ["/start.sh"]