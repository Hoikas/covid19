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
import concurrent.futures
from functools import partial
import itertools
import gzip
import json
import logging
from pathlib import Path
import sys
import tempfile
import urllib.request

_parser = argparse.ArgumentParser(description="COVID-19 Graph")
_parser.add_argument("-c", "--config", type=Path, help="path to configuration file", default="config.ini")
_sub_parsers = _parser.add_subparsers(title="command", dest="command", required=True)

# Dump Generated Census command
_dump_parser = _sub_parsers.add_parser("dumpjson")
_dump_parser.add_argument("-p", "--pretty", action="store_true", help="pretty format json", default=False)
_dump_parser.add_argument("--no-zerofill", action="store_true", help="don't zerofill counties with no cases", default=False)
_dump_parser.add_argument("dest", type=Path, help="path to dump generated json", nargs="?", default="out")

# Graph command
_graph_parser = _sub_parsers.add_parser("graph")
_graph_parser.add_argument("-d", "--data", type=str, help="path to json data")
_graph_parser.add_argument("-z", "--zkey", type=str, help="z key of data to graph (omit for all)")
_graph_parser.add_argument("dest", help="path to output the graph (will open in browser if omitted", nargs="?", default="")

# Minify
_minify_parser = _sub_parsers.add_parser("minifygeojson")
_minify_parser.add_argument("dest", type=Path, help="path to dump minified geojson")

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
}

CAPITA = 1000

def _download_csv(**kwargs):
    logging.info("Downloading CSV files...")

    result = {}
    for key, url in kwargs.items():
        if not url:
            logging.warning(f"Omitting data source '{key}' -- hope you know what you're doing...")
            result[key] = None
            continue

        logging.debug(f"Downloading {url}")
        with urllib.request.urlopen(url) as response:
            fp = tempfile.SpooledTemporaryFile(mode="w+")
            fp.write(response.read().decode("windows-1252"))
            fp.seek(0)
            result[key] = fp
            logging.debug(f"... done with {url}")
    return result

def _fetch_csv(config):
    # Pull down the required remote CSV files
    csv_files = _download_csv(census=config["data"]["uscensus"],
                              ecdc=config["data"]["ecdc"],
                              nyt_county=config["data"]["covid19"])
    # original is out-of-date, so use in-repo copy
    csv_files["fips"] = open(FIPS_PATH, "r", encoding="windows-1252")
    return csv_files

def _write_output(dest_path, contents, compression=None):
    if isinstance(contents, str):
        contents = contents.encode("utf-8")
    assert isinstance(contents, (bytes, bytearray))

    dest_path.parent.mkdir(exist_ok=True, parents=True)
    if compression is None:
        with dest_path.open("wb") as fp:
            fp.write(contents)
    elif compression == "gz":
        gz_path = dest_path.with_suffix(f"{dest_path.suffix}.gz")
        with gz_path.open("wb") as fp:
            with gzip.GzipFile(filename=dest_path.name, fileobj=fp, mode="wb") as gzfp:
                gzfp.write(contents)
    elif compression == "br":
        # Maybe you can has brotli?
        br_path = dest_path.with_suffix(f"{dest_path.suffix}.br")
        try:
            import brotli
        except ImportError:
            logging.warning(f"{br_path.name} not produced (brotli failed to import)")
        else:
            with br_path.open("wb") as fp:
                fp.write(brotli.compress(contents))
    else:
        raise RuntimeError(compression)

def _generate_data(out_path, ecdc_file, census_file, fips_file, nytimes_file, pretty=False, concise=False):
    import csv
    from collections import defaultdict

    logging.info("Generating data...")

    # Take the US census estimate for 2019 and calculate per-capita statistics.
    # TODO: when the 2020 census is complete, that might be a better source of data.
    logging.debug("Digging through Census data...")
    census = csv.DictReader(census_file)
    fips = csv.DictReader(fips_file)

    # Known mismatched lines due to turds like encodings (damn you windows-1252)
    known_mismatches = { 1834: "Doña Ana County" }

    # Apply the population from the census to each FIPS code
    fips_info = {}
    for i, (i_census, i_fips) in enumerate(zip(census, fips)):
        # make sure the stuff matches. hopefully no spelling errors are in the files...
        # damn that Doña Ana County... and damn that Microsoft windows-1252 encoding.
        census_county_name, fips_name = i_census["CTYNAME"].strip(), i_fips["name"].strip()
        if census_county_name.lower() != fips_name.lower():
            census_county_name = known_mismatches.get(i)
            if census_county_name is None:
                logging.critical(f"Census/FIPS CSV mismatch: {i} {census_name} || {fips_name}")
                raise RuntimeError()

        # State population is given when CTYNAME == STNAME. Stash as both the state name
        # and the fips state code for ease of use.
        census_state_name = i_census["STNAME"].strip()
        if census_county_name == census_state_name:
            title = census_state_name
            the_fips_code = int(i_fips["fips"].strip().zfill(5)[:-3])
            location = i_fips["state"].strip()
            # include: state name, state fips code, county-link fips code
            fips_codes = [census_state_name, the_fips_code, int(i_fips["fips"].strip().zfill(5))]
            state = True
        else:
            title = f"{census_county_name}, {census_state_name}"
            location = int(i_fips["fips"])
            fips_codes = [location]
            state = False

        pop = int(i_census["POPESTIMATE2019"])
        for fips_code in fips_codes:
            fips_info[fips_code] = { "location": location,
                                     "population": pop,
                                     "state": state,
                                     "title": title }


    # OK, so before we can calculate the per-capita, some of the cases/deaths were reported by multiple
    # FIPS codes. For example, "New York City" was listed w/o a FIPS code but New York City is made of five counties...
    # So, we need to merge the total population of those fips codes
    fips_merge = {}
    for (city, state), i in COUNTY_REMAP.items():
        for fips_code in i:
            fips_merge[fips_code] = { "population": sum((fips_info[j]["population"] for j in i)),
                                      "state": False,
                                      "title": f"{city}, {state}" }
    fips_info.update(fips_merge)


    # Load all NYT data into memory so we can operate on it later
    # Format: [date][fipscode][...]
    logging.debug("Reading from New York Times CSV...")
    nyt = csv.DictReader(nytimes_file)

    # dict of dicts of dicts of ints
    data = defaultdict(partial(defaultdict, partial(defaultdict, int)))

    for i in nyt:
        state_name = i["state"].strip()
        if state_name not in fips_info:
            # Territories are not listed in the Census data... wtf?
            continue

        fips = i["fips"].strip()
        if not fips:
            fips = COUNTY_REMAP.get((i["county"], i["state"]))
            state = False
        else:
            fips = int(fips)
            state = fips_info[fips]["state"]
            fips = [fips,]

        date_data = data[i["date"].strip()]
        if fips:
            for fips_code in fips:
                county_info = fips_info[fips_code]
                county_datum = date_data[fips_code]
                # Plotly seems to have issues with 4 digit county codes
                county_datum["location"] = str(fips_code).zfill(2 if state else 5)
                county_datum["population"] = county_info["population"]
                county_datum["title"] = county_info["title"]
                county_datum["total_cases"] += int(i["cases"])
                county_datum["total_deaths"] += int(i["deaths"])
        elif i["county"].lower() != "unknown":
            # don't whine on "unknown" - we know.
            logging.warning(f"No FIPS code for datum {i}")

        state_info = fips_info[state_name]
        state_datum = date_data[state_name]
        state_datum["title"] = state_name
        state_datum["population"] = state_info["population"]
        state_datum["location"] = state_info["location"]
        state_datum["total_cases"] += int(i["cases"])
        state_datum["total_deaths"] += int(i["deaths"])

    # Zero fill
    if not concise:
        logging.debug("Zero filling counties with no cases...")
        all_locations = frozenset((value["title"] if value["state"] else key for key, value in fips_info.items()))
        for date_data in data.values():
            our_locations = frozenset(date_data.keys())
            for i in all_locations - our_locations:
                info = fips_info[i]
                zerofill = date_data[i]
                if info["state"]:
                    zerofill["location"] = info["location"]
                else:
                    zerofill["location"] = str(i).zfill(5)

                zerofill["population"] = info["population"]
                zerofill["title"] = info["title"]
                zerofill["total_cases"] = 0
                zerofill["total_deaths"] = 0

    # Add in optional international data
    if ecdc_file:
        logging.debug("Loading international data...")
        ecdc = csv.DictReader(ecdc_file)

        # Bad news, old chap. This data is per-day, not cumulative. Therefore, we will need
        # to iterate through it twice since it is stored backwards. Sigh...
        countries = set()
        for i in ecdc:
            # oddly, the EU does not use ISO 8601
            date = f"{i['year']}-{i['month'].zfill(2)}-{i['day'].zfill(2)}"
            if date not in data:
                # don't want any data for when no US detail is available
                continue

            iso3 = i["countryterritoryCode"]
            countries.add(iso3)
            datum = data[date][iso3]
            datum["location"] = iso3
            # Some countries don't have population data? Interesting.
            if i["popData2018"].isnumeric():
                datum["population"] = int(i["popData2018"])
            datum["title"] = i["countriesAndTerritories"].replace('_', ' ')
            datum["total_cases"] = int(i["cases"])
            datum["total_deaths"] = int(i["deaths"])

        counter = defaultdict(partial(defaultdict, int))
        for date_data in (data[date] for date in sorted(data)):
            for datum, country_counter in ((date_data[country], counter[country]) for country in countries
                                                                                  if country in date_data):
                country_counter["total_cases"] += datum["total_cases"]
                country_counter["total_deaths"] += datum["total_deaths"]
                datum["total_cases"] = country_counter["total_cases"]
                datum["total_deaths"] = country_counter["total_deaths"]

    logging.debug("Writing JSON to disk...")
    for date, date_data in data.items():
        # For now, state and county data are dumped to the same json file.
        with out_path.joinpath(f"{date}.json").open("w") as fp:
            indent = 4 if pretty else None
            json.dump(list(date_data.values()), fp, indent=indent)


def _dump_json(args, config):
    out_path = args.dest
    if out_path.exists() and not out_path.is_dir():
        logging.critical(f"Specified output path must be a directory: {outpath}")
        sys.exit(1)

    csv_files = _fetch_csv(config)
    out_path.mkdir(parents=True, exist_ok=True)
    _generate_data(out_path, csv_files["ecdc"], csv_files["census"], csv_files["fips"],
                   csv_files["nyt_county"], args.pretty, args.no_zerofill)


def _graph(args, config):
    logging.info("Starting graph command!")
    if args.data:
        _generate_graph(args, config, Path(args.data))
    else:
        csv_files = _fetch_csv(config)
        with tempfile.TemporaryDirectory(prefix="covid19") as td:
            json_path = Path(td)
            _generate_data(json_path, csv_files["ecdc"], csv_files["census"], csv_files["fips"], csv_files["nyt_county"])
            _generate_graph(args, config, json_path)

def _generate_graph(args, config, data_path):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    all_subplots = [
        dict(title=f"Cases Per {CAPITA:,} Population", zkey="cases_per_capita", use_std=True, row=1, col=1),
        dict(title="Total Cases (log10)", zkey="log_cases", use_std=False, row=1, col=2),
        dict(title=f"Deaths Per {CAPITA:,} Population", zkey="deaths_per_capita", use_std=True, row=2, col=1),
        dict(title="Total Deaths (log10)", zkey="log_deaths", use_std=False, row=2, col=2),
    ]
    map_configs = {
        "Show US Counties": dict(geojson=config["map"]["county_geojson"], dkey="county"),
        "Show US States": dict(geojson=config["map"]["state_geojson"], dkey="states"),
        "Show Countries": dict(geojson=config["map"]["world_geojson"], dkey="world"),
    }

    # Overall layout looks like this:
    # (map: cases per capita) (map: log total cases)
    # (map: deaths per capita) (map: log total deaths)
    # (annotations)
    # (slider control)
    all_figs = make_subplots(rows=2, cols=2, vertical_spacing=0.05,
                             specs=[[{"type": "choroplethmapbox"}, {"type": "choroplethmapbox"}],
                                   [{"type": "choroplethmapbox"}, {"type": "choroplethmapbox"}]])

    # Pick which subplots we want to render... Note that if we are not generating the HTML, then
    # we should only generate ONE.
    subplots = []
    if args.zkey:
        subplot = next((i for i in all_subplots if i["zkey"].lower() == args.zkey), None)
        if subplot is None:
            logging.critical(f"invalid zkey specified: '{args.zkey}'")
            return
        subplots.append((subplot["zkey"], None, [subplot]))
    else:
        subplots.append(("all_choropleths", all_figs, all_subplots))
        if args.dest:
            subplots.extend((i["zkey"], None, [i]) for i in all_subplots)

    # Do data stuff now that we know we won't fail on stupid crap.
    data = list(_generate_graph_data(data_path))

    default_map = next(iter(map_configs.values()))
    figures = {}
    for fig_name, fig, fig_subplots in subplots:
        for subplot in fig_subplots:
            fig = _generate_choropleth_traces(fig, data, default_map["geojson"], default_map["dkey"], **subplot)
        _add_fig_controls(fig, data, map_configs, fig_subplots)
        figures[fig_name] = fig

    compression = (None, "gz", "br")
    if args.dest:
        logging.debug("Writing Figures...")
        output_path = Path(args.dest)
        if len(figures) == 1:
            logging.debug(f"Writing HTML... {output_path}")
            html = next(iter(figures.values())).to_html(include_plotlyjs="cdn")
            with concurrent.futures.ProcessPoolExecutor() as executor:
                for i in comression:
                    executor.submit(_write_output, output_path, html, i)
        else:
            logging.debug("Generating HTML...")
            with concurrent.futures.ProcessPoolExecutor() as executor:
                to_html = partial(go.Figure.to_html, include_plotlyjs="cdn")
                for zkey, html in zip(figures.keys(), executor.map(to_html, figures.values())):
                    fig_path = output_path.joinpath(f"{zkey}.html")
                    logging.debug(f"Writing HTML... {fig_path.name}")
                    for i in compression:
                        executor.submit(_write_output, fig_path, html, i)

    else:
        logging.debug("Showing figure...")
        assert len(figures) == 1
        next(iter(figures.values())).show()

def _generate_graph_data(data_path):
    from collections import namedtuple
    from math import isnan, log10
    import pandas as pd

    data_entry = namedtuple("data_entry", ["date", "df", "hovertemplate", "stats"])
    make_hover = lambda x: f"<b>{x['title']}</b><br>" \
                           f"Population: {x['population']:,.00f}<br>"  \
                           f"Total Cases: {x['total_cases']:,}<br>" \
                           f"Cases Per {CAPITA:,} Population: {x['cases_per_capita']:,.04}<br>" \
                           f"Total Deaths: {x['total_deaths']:,}<br>" \
                           f"Deaths Per {CAPITA:,} Population: {x['deaths_per_capita']:,.04}<br>" \
                           f"Fatality Rate: {x['fatality_rate']:.02%}"
    apply_log = lambda value: 0 if value <= 0 else log10(value)

    def make_stats_dict(*args, **kwargs):
        stats_entry = namedtuple("stats_entry", ["max", "mean", "median", "min", "std"])
        make_stats = lambda x: stats_entry(x.max(), x.mean(), x.median(), x.min(), x.std())
        return { i: { j: make_stats(frame[j]) for j in args } for i, frame in kwargs.items() }

    # Each JSON file in the data path is a trace on the figure. The slider will allow us to select
    # which trace the user is viewing.
    logging.info("Loading graph data frames..")
    for i in sorted(data_path.glob("*.json")):
        logging.debug(f"Reading {i.name}...")
        df = pd.read_json(i, dtype={"location": False})
        logging.debug("... Processing")

        # Total numbers are exponential, so create log columns for that data
        df["log_cases"] = df["total_cases"].apply(apply_log)
        df["log_deaths"] = df["total_deaths"].apply(apply_log)

        # Moved rate calculations here for ease of use.
        df["cases_per_capita"] = df.apply(lambda row: row["total_cases"] / (row["population"] / CAPITA), axis=1)
        df["deaths_per_capita"] = df.apply(lambda row: row["total_deaths"] / (row["population"] / CAPITA), axis=1)
        df["fatality_rate"] = df.apply(lambda row: 0 if row["total_cases"] == 0 else
                                                   row["total_deaths"] / row["total_cases"],
                                       axis=1)

        # I apologize in advance for this sin.
        df["text"] = df.apply(make_hover, axis=1)

        # Segregate state and county values into their own data frames to prevent statistics duplication.
        # Rules: states and international countries are non-numeric strings
        #        counties are numeric strings... Note... including ISO-3 country codes as well...
        states_df = df.query("location.str.isalpha()")
        county_df = df.query("location.str.isnumeric() or location.str.len() == 3")
        world_df = df.query("location.str.isalpha() and location.str.len() == 3")

        # Some counties are duplicated due to the way the NYT reports the data.
        # Keeping this separate incase we start drawing scattermaps.
        county_dedup_df = county_df.drop_duplicates("title", inplace=False)

        # Nuke international countries for stats
        county_stats_df = county_dedup_df.query("location.str.isnumeric()")
        states_stats_df = states_df.query("location.str.len() == 2")
        stats = make_stats_dict("cases_per_capita", "deaths_per_capita", "log_cases", "log_deaths", "total_cases",
                                county=county_stats_df, states=states_stats_df, world=world_df)

        hovertemplate = "%{text}" \
                        f"<extra><b>US Case Data for {i.stem}</b><br>" \
                        f"Mean Cases Per {CAPITA:,} Population: {stats['county']['cases_per_capita'].mean:,.04}<br>" \
                        f"Median Cases Per {CAPITA:,} Population: {stats['county']['cases_per_capita'].median:,.04}<br>" \
                        f"STD: {stats['county']['cases_per_capita'].std}<br>" \
                        f"Mean Total Cases Per County: {stats['county']['total_cases'].mean:,.04}<br>" \
                        f"Median Total Cases Per County: {stats['county']['total_cases'].median:,.04}<br>" \
                        f"STD: {stats['county']['total_cases'].std}</extra>"

        # NOTE: reusing the same data frames so we can just swap out the geojson for different views.
        #       all the above chicanery was for the statistics.
        yield data_entry(i.stem, dict(county=df, states=df, world=df), hovertemplate, stats)

def _add_fig_controls(fig, data, map_configs, subplots):
    logging.debug("Generating misc layout...")
    fig.update_layout(annotations=[dict(x=0.50, y=1.1,
                                        xref="paper", yref="paper",
                                        yanchor="top",
                                        text="COVID-19 Spread Graphs by <a href='mailto:AdamJohnso@gmail.com'>Adam Johnson</a>. " \
                                             "<a href='https://github.com/Hoikas/covid19'>(Source)</a><br>" \

                                             "US Data from <a href='https://www.nytimes.com/interactive/2020/us/coronavirus-us-cases.html'>" \
                                             "The New York Times</a>, based on reports from state and local health agencies. " \
                                             "<a href='https://github.com/nytimes/covid-19-data'>(Source)</a><br>" \

                                             "World Data from the <a href='https://www.ecdc.europa.eu/en'>" \
                                             "European Centre for Disease Prevention and Control</a>. " \
                                             "<a href='https://www.ecdc.europa.eu/en/publications-data/download-todays-data-geographic-distribution-covid-19-cases-worldwide'>" \
                                             "(Source)</a>",
                                        showarrow=False)])

    # I had to change from a choropleth to a choroplethmapbox. Despite the poor documentation of plotly,
    # the difference appears to the that the former uses a webgl powered service known as mapbox.
    # This handles the county geojson much better than the builtin map, though sadly the builtin
    # map lets us zoom in exclusively on the US. If we ever go back to choropleth, set `geo_scope="usa"`
    fig.update_mapboxes(style="carto-positron",
                        center={"lat": 37.0902, "lon": -95.7129},
                        zoom=3)

    # Map type dropdown ahoy!
    map_buttons = []
    for name, map_config in map_configs.items():
        button = dict(label=name, method="restyle", args=[dict(geojson=map_config["geojson"])])
        map_scales = [_make_scale(datum.stats[map_config["dkey"]][subplot["zkey"]], subplot["use_std"])
                      # Danger: outer -> inner
                      for subplot in subplots for datum in data]
        button["args"][0]["zmin"] = [i[0] for i in map_scales]
        button["args"][0]["zmax"] = [i[1] for i in map_scales]
        map_buttons.append(button)
    fig.update_layout(updatemenus=[dict(buttons=map_buttons,
                                        direction="down",
                                        showactive=True,
                                        pad=dict(r=10, t=10),
                                        x=0.0, y=1.1,
                                        xanchor="left", yanchor="top")])

    # Slider control
    def is_trace_visible(slider_idx, trace_idx):
        # multiple traces are visible for each slider entry...
        return slider_idx == trace_idx % len(data)

    for i, fig_datum in enumerate(fig.data):
        # ugh
        fig_datum.visible = is_trace_visible(len(data)-1, i)
    date_steps = [dict(method="restyle",
                       args=["visible", [is_trace_visible(i, j) for j in range(len(fig.data))]],
                       label=data[i].date) for i, date in enumerate(data)]
    fig.update_layout(sliders=[dict(active=len(data)-1,
                                    steps=date_steps,
                                    currentvalue=dict(prefix="Showing COVID-19 Data for "))])

def _make_scale(stats, use_std):
    if use_std:
        mean = stats.mean
        stddev = stats.std
        zmax = mean + stddev
        zmin = max(mean-stddev, 0.0)
    else:
        zmax = stats.max
        # Prevents a negative scale from appearing
        if zmax == 0.0:
            zmax = 1.0
        zmin = stats.min
    return zmin, zmax

def _generate_choropleth_traces(fig, data, geojson, dkey, /, *, title, zkey, row, col, use_std=True):
    import plotly.graph_objects as go
    logging.debug(f"Generating choropleths for '{title}'")

    # Doggone color bars just appear whereever they please :/
    # Unfortunately, the plotly documentation is vague with NO examples about how to position
    # these ruddy things. This is my best guesswork. If you can fix it to not suck, be my guest...
    colorbars = {
        (1, 1): dict(x=-0.05, y=0.774, len=0.51),
        (1, 2): dict(x=0.50, y=0.774, len=0.51),
        (2, 1): dict(x=-0.05, y=0.248, len=0.51),
        (2, 2): dict(x=0.50, y=0.248, len=0.51),
    }

    if fig is None:
        fig, colorbar = go.Figure(), {}
        row, col = None, None
    else:
        colorbar = colorbars[(row, col)]
    colorbar["title"] = title

    traces = []
    for i, datum in enumerate(data):
        df = datum.df[dkey]
        zmin, zmax = _make_scale(datum.stats[dkey][zkey], use_std)
        traces.append(dict(type="choroplethmapbox",
                           geojson=geojson,
                           name=datum.date,
                           locations=df["location"],
                           z=df[zkey],
                           zmax=zmax, zmin=zmin,
                           text=df["text"],
                           hovertemplate=datum.hovertemplate,
                           colorbar=colorbar,
                           marker_opacity=0.5, marker_line_width=0,
                           # Portland and Temps offer the best visualizations IME
                           colorscale="Portland"))

    # This minimizes the plotly API calls and is MUCH faster.
    rows = [row] * len(data) if row is not None else None
    cols = [col] * len(data) if col is not None else None
    fig.add_traces(traces, rows, cols)
    return fig


def _minify(src_path, dest_path, compression=None):
    with src_path.open("r") as fp:
        geo = json.load(fp)
    for i in geo["features"]:
        if "properties" in i:
            del i["properties"]
    _write_output(dest_path, json.dumps(geo, separators=(',', ':')), compression)

def _minify_geojson(args, config):
    logging.info("Minifying geojson...")

    data_path = Path(__file__).parent.joinpath("data")
    geojson_iter = (i for i in data_path.glob("*.json") if i.name.startswith("geojson"))
    compression = (None, "gz", "br")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i in geojson_iter:
            for j in compression:
                executor.submit(_minify, i, args.dest.joinpath(i.name), j)


if __name__ == "__main__":
    args = _parser.parse_args()

    logging.basicConfig(format="[%(asctime)s] %(levelname)s: %(message)s", level=logging.DEBUG)
    logging.info("__main__")

    from configparser import ConfigParser
    config = ConfigParser()
    config.read(args.config)

    if args.command == "dumpjson":
        _dump_json(args, config)
    elif args.command == "graph":
        _graph(args, config)
    elif args.command == "minifygeojson":
        _minify_geojson(args, config)
    else:
        raise RuntimeError()

    logging.info("Have a nice day!")
