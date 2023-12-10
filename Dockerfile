FROM python:3.11.7

WORKDIR /app
COPY requirements.txt /app
COPY data_acq.py /app
COPY data_prep.py /app
COPY node2vec.py /app
COPY GNN.py /app
COPY vgae.py /app
COPY run_baselines.py /app
COPY run_GNN.py /app
COPY run_vgae.py /app
COPY plot_results.py /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

RUN mkdir /app/data
RUN mkdir /app/graphs
RUN mkdir /app/plots
RUN mkdir /app/scores

CMD ["bash", "-c", "python data_acq.py && python data_prep.py && python run_baselines.py && python run_GNN.py && python run_vgae.py && python plot_results.py"]