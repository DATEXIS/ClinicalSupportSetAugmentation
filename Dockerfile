FROM registry.datexis.com/pgrundmann/pytorch-ngc:22.04
RUN pip3 install numpy rich pandas transformers tqdm pytorch-lightning \
                 fastapi uvicorn[standard] jsonargparse[signatures] \
                 deepspeed gpustat pyarrow  tensorboardX \
                 torchmetrics wandb optuna psycopg2-binary accelerate datasets
WORKDIR /src
COPY src .

CMD ["/usr/sbin/sshd", "-D"]
