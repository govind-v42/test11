FROM python:3.10.6-buster as pipeline


RUN mkdir dvc_pipeline/data
COPY data/data2.csv dvc_pipeline/data/data2.csv


COPY requirements.txt dvc_pipeline/requirements.txt 
RUN pip install -r requirements.txt

COPY dvc.yaml dvc_pipeline/dvc.yaml

WORKDIR /dvc_pipeline
RUN dvc init --no-scm
RUN dvc repro


#STAGE 2 docker: webapp

FROM python:3.10.6-buster

RUN mkdir web_app

COPY --from=pipeline app.py web_app/app.py
COPY --from=pipeline finalized_model.pkl web_app/finalized_model.pkl

COPY requirements.txt   web_app/requirements.txt
RUN pip install -r requirements.txt

WORKDIR /web_app
EXPOSE 5000
CMD ["python", "app.py"]

RUN pip install flask
