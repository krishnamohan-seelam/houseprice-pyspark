FROM jupyter/pyspark-notebook:latest
WORKDIR /work/
COPY ./requirements.txt /work/
RUN pip install -r requirements.txt
COPY ./house-prices /work/house-prices
COPY ./model.py /work/
COPY ./pipeline_model.h5 /work/pipeline_model.h5
COPY ./lr_model.h5 /work/lr_model.h5
ENTRYPOINT ["spark-submit","model.py"]
