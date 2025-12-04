import importlib
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True, help="Which experiment module to run")
    args, unknown = parser.parse_known_args()  # capture extra args for the experiment

    # Dynamically import the experiment module
    module_name = f"softprompt_experiments.experiments.{args.experiment}"
    exp_module = importlib.import_module(module_name)

    if not hasattr(exp_module, "run"):
        raise ValueError(f"Experiment '{args.experiment}' must have a run(args) function.")

    # Forward unknown args to the experiment's own parser
    exp_module.run(unknown)

if __name__ == "__main__":
    main()
