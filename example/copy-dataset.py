import hub
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--src', help='Source dataset.')
parser.add_argument('-d', '--dst', help='Destination dataset.')
args = parser.parse_args()

hub.copy(args.src, args.dst)
