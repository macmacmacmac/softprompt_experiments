import importlib
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Name of experiment module")
    args = parser.parse_args()

    module_name = f"softprompt_experiments.experiments.{args.experiment}"

    try:
        exp = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ValueError(f"Experiment '{args.experiment}' not found.")

    if not hasattr(exp, "run"):
        raise ValueError(f"Experiment '{args.experiment}' has no run() function.")

    exp.run()

if __name__ == "__main__":
    main()
