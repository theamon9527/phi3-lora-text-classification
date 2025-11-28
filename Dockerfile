FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime
WORKDIR /app
COPY . /app
RUN pip install --upgrade pip \
    && pip install -r requirements.txt
EXPOSE 8000
CMD ["python", "api_service.py"]
