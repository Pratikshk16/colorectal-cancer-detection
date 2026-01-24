from kfp import dsl
import kfp

def data_processing_operation():
    return dsl.ContainerOp(
        name="Data Processing",
        image="pratikshk/mlops-app:latest",
        command=["python", "src/data_processing.py"],
    )

def model_training_operation():
    return dsl.ContainerOp(
        name="Model Training",
        image="pratikshk/mlops-app:latest",
        command=["python", "src/model_training.py"],
    )

@dsl.pipeline(
    name="MLOPS Pipeline",
    description="This is my first Kubeflow Pipeline",
)
def mlops_pipeline():
    dp = data_processing_operation()
    mt = model_training_operation().after(dp)

if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        mlops_pipeline, "mlops_pipeline.yaml"
    )
