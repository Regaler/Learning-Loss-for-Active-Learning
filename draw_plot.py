import os
import argparse
import matplotlib.pyplot as plt


def draw_single(result_path, label):
    """
    Draw the performance by dataset size plot
    typically used for active learning research

    Parameters
    ----------
    result_path: str
        the txt result file from AL experiment
        it is a series of numbers delimited by "\n"
    label: str
        The label for this curve

    Returns
    -------
    str
        the resultant picture path
    """
    with open(result_path, "r") as f:
        data = f.readlines()
    data = [float(x) for x in data]
    plt.plot(range(len(data)), data, label=label)


def draw(results, title="Active_Learning_on_CIFAR10"):
    """
    Draw the performance by dataset size plot
    for multiple results overlayed together

    Parameters
    ----------
    results: list[str]
        the list of result paths
    """
    def _parse(path):
        # ex) results/results_LL4AL_0.txt -> results_LL4AL_0
        return os.path.splitext(os.path.basename(path))[0]

    for result in results:
        draw_single(result, label=_parse(result))
    plt.title(title.replace("_", " "))
    plt.legend()
    plt.savefig(f"results/{title}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--title', type=str,
                        help='The title of the figure')
    parser.add_argument('--result_paths', type=str, nargs="+",
                        help='The paths to the result txt files')
    args = parser.parse_args()
    paths = args.result_paths[0].split(",")
    draw(paths, title=args.title)
