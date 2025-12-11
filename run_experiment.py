import importlib
import argparse
import sys

def main():
    print("\nReceived the following arguments:\n|---->", " ".join(sys.argv))
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        nargs="+",                   # <── allow multiple
        required=True,
        help="Experiment modules to run"
    )
    args, unknown = parser.parse_known_args()

    for exp in args.experiment:
        module_name = f"softprompt_experiments.experiments.{exp}"
        exp_module = importlib.import_module(module_name)

        if not hasattr(exp_module, "run"):
            raise ValueError(f"Experiment '{exp}' must have a run(args) function.")

        # Each experiment receives same unknown args (or you could customize)
        exp_module.run(unknown)


if __name__ == "__main__":
    main()
