#    This file is part of COVID-19 Graph
#
#    COVID-19 Graph is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    COVID-19 Graph is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with COVID-19 Graph.  If not, see <http://www.gnu.org/licenses/>.

import argparse
import itertools
import logging
from pathlib import Path
import sys

_parser = argparse.ArgumentParser(description="COVID-19 Graph")
_parser.add_argument("-c", "--config", type=Path, help="path to configuration file", default="config.ini")
_sub_parsers = _parser.add_subparsers(title="command", dest="command", required=True)

# Dump Generated Census command
_dump_parser = _sub_parsers.add_parser("dumpjson")
_dump_parser.add_argument("-p", "--pretty", action="store_true", help="pretty format json", default=False)
_dump_parser.add_argument("dest", type=Path, help="path to dump generated json", nargs="?", default="out")

# FIPS codes for dealing with the Census Bureau's nonsense--updated for 2019 estimate
# original source: https://github.com/kjhealy/fips-codes
FIPS_PATH = Path(__file__).parent.joinpath("state_and_county_fips_master.csv")

# The NYT data sometimes reports cases for a certain area instead of a FIPS code. In that case,
# we need to remap that area to the FIPS codes it encompasses. See the README.md in the NYT
# repository, subsection "Geographic Exceptions"
# NOTE: I see no need to merge Chicago's Counties due to the geographic situation.
COUNTY_REMAP = {
    ("Joplin", "Missouri"): (29097, 29145),
    ("Kansas City", "Missouri"): (29037, 29047, 29095, 29165),
    ("New York City", "New York"): (36005, 36047, 36061, 36081, 36085),

    # !!! undocumented turd !!!
    ("Dukes and Nantucket", "Massachusetts"): (25007, 25019),
}

def _download_csv(**kwargs):
    import tempfile
    import urllib.request

    logging.info("Downloading CSV files...")

    result = {}
    for key, url in kwargs.items():
        logging.debug(f"Downloading {url}")
        with urllib.request.urlopen(url) as response:
            fp = tempfile.SpooledTemporaryFile(mode="w+")
            fp.write(response.read().decode("windows-1251"))
            fp.seek(0)
            result[key] = fp
            logging.debug(f"... done with {url}")
    return result

def _generate_data(out_path, census_file, fips_file, nytimes_file, pretty=False):
    import csv
    from collections import defaultdict
    from functools import partial
    import json

    logging.info("Generating data...")

    # Load all NYT data into memory so we can operate on it later
    # Format: [date][fipscode][...]
    logging.debug("Reading from New York Times CSV...")
    nyt = csv.DictReader(nytimes_file)

    # dict of dicts of dicts of ints
    data = defaultdict(partial(defaultdict, partial(defaultdict, int)))

    for i in nyt:
        fips = i["fips"].strip()
        if not fips:
            fips = COUNTY_REMAP.get((i["county"], i["state"]))
        else:
            fips = [int(fips)]

        date_data = data[i["date"].strip()]
        if fips:
            for fips_code in fips:
                county_datum = date_data[fips_code]
                county_datum["total_cases"] += int(i["cases"])
                county_datum["total_deaths"] += int(i["deaths"])
        elif i["county"].lower() != "unknown":
            # don't whine on "unknown" - we know.
            logging.warning(f"No FIPS code for datum {i}")

        state_datum = date_data[i["state"].strip()]
        state_datum["total_cases"] += int(i["cases"])
        state_datum["total_deaths"] += int(i["deaths"])


    # Take the US census estimate for 2019 and calculate per-capita statistics.
    # TODO: when the 2020 census is complete, that might be a better source of data.
    logging.debug("Digging through Census data...")
    census = csv.DictReader(census_file)
    fips = csv.DictReader(fips_file)

    # Known mismatched lines due to turds like encodings (damn you windows-1252)
    known_mismatches = (1834,)

    # Apply the population from the census to each FIPS code
    fips_population = {}
    for i, (i_census, i_fips) in enumerate(zip(census, fips)):
        # make sure the stuff matches. hopefully no spelling errors are in the files...
        # damn that Doña Ana County... and damn that Microsoft windows-1252 encoding.
        census_name, fips_name = i_census["CTYNAME"], i_fips["name"]
        if census_name.lower() != fips_name.lower() and i not in known_mismatches:
            logging.critical(f"Census/FIPS CSV mismatch: {i} {census_name} || {fips_name}")
            raise RuntimeError()
        fips_population[int(i_fips["fips"])] = int(i_census["POPESTIMATE2019"])

        # State population is given when CTYNAME == STNAME. These are not FIPS codes, but
        # we are storing state data keyed by state name, so there you go.
        if i_census["CTYNAME"].strip() == i_census["STNAME"].strip():
            fips_population[i_census["STNAME"].strip()] = int(i_census["POPESTIMATE2019"])

    # OK, so before we can calculate the per-capita, some of the cases/deaths were reported by multiple
    # FIPS codes. For example, "New York City" was listed w/o a FIPS code but New York City is made of five counties...
    # So, we need to merge the total population of those fips codes
    fips_population_merge = {}
    for i in COUNTY_REMAP.values():
        for fips_code in i:
            fips_population_merge[fips_code] = sum((fips_population[j] for j in i))
    fips_population.update(fips_population_merge)

    # Calculate per capita
    logging.debug("Calculating per capita rates...")
    for location, datum in itertools.chain.from_iterable((i.items() for i in data.values())):
        pop = fips_population.get(location)
        if pop:
            datum["population"] = pop
            datum["cases_per_capita"] = datum["total_cases"] / pop
            datum["deaths_per_capita"] = datum["total_deaths"] / pop
            datum["mortality"] = datum["deaths_per_capita"] / datum["cases_per_capita"]
        else:
            logging.warning(f"No population available for {location}")

    logging.debug("Writing JSON to disk...")
    for date, date_data in data.items():
        # For now, state and county data are dumped to the same json file.
        with out_path.joinpath(f"{date}.json").open("w") as fp:
            indent = 4 if pretty else None
            json.dump(date_data, fp, indent=indent)

def _dump_json(args, config, csv_files):
    out_path = args.dest
    if out_path.exists() and not out_path.is_dir():
        logging.critical(f"Specified output path must be a directory: {outpath}")
        sys.exit(1)

    out_path.mkdir(parents=True, exist_ok=True)
    _generate_data(out_path, csv_files["census"], csv_files["fips"], csv_files["nyt_county"], args.pretty)


if __name__ == "__main__":
    args = _parser.parse_args()

    logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.DEBUG)
    logging.info("__main__")

    from configparser import ConfigParser
    config = ConfigParser()
    config.read(args.config)

    # Pull down the required remote CSV files
    csv_files = _download_csv(census=config["data"]["uscensus"],
                              nyt_county=config["data"]["covid19"])
    # original is out-of-date, so use in-repo copy
    csv_files["fips"] = open(FIPS_PATH, "r")

    if args.command == "dumpjson":
        _dump_json(args, config, csv_files)
    else:
        raise RuntimeError()
