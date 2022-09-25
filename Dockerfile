FROM python:3.10.6-buster as pipeline



RUN mkdir dvc_pipeline

COPY data/data2.csv dvc_pipeline/data/data2.csv

COPY etlpipeline.py dvc_pipeline/etlpipeline.py


COPY traintest.py dvc_pipeline/traintest.py

COPY app.py   dvc_pipeline/app.py

# COPY app.py dvc_pipeline/app.py

COPY requirements.txt dvc_pipeline/requirements.txt 

 

COPY dvc.yaml dvc_pipeline/dvc.yaml


WORKDIR /dvc_pipeline

RUN pip install -r requirements.txt




RUN dvc init --no-scm
RUN dvc repro


# STAGE 2 docker: webapp

FROM python:3.10.6-buster

RUN  mkdir -p web_app

COPY --from=pipeline finalized_model1.pkl web_app/finalized_model1.pkl

 
COPY  --from=pipeline feature1.pkl  web_app/feature1.pkl

COPY  app.py web_app/app.py
COPY  templates web_app/templates



COPY requirements.txt   web_app/requirements.txt

WORKDIR /web_app
RUN pip install -r requirements.txt


EXPOSE 5000
CMD ["python", "app.py"]

# RUN pip install flask
