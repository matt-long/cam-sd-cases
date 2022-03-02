import os
import copy
from subprocess import check_call, Popen, PIPE
from collections import OrderedDict

import warnings

import numpy as np
import xarray as xr

molecular_weights = dict(
    air=28.966,
    O2=32.0,
    CO2=44.0,
    N2=28.0,
)


chem_mech_template = OrderedDict(
    [
        ("header", []),
        (
            "species",
            OrderedDict(
                [
                    ("solution", []),
                    ("fixed", []),
                    ("col-int", []),
                    ("not-transported", []),
                ]
            ),
        ),
        (
            "solution classes",
            OrderedDict(
                [
                    ("explicit", []),
                    ("implicit", []),
                ]
            ),
        ),
        (
            "chemistry",
            OrderedDict(
                [
                    ("photolysis", []),
                    ("reactions", []),
                    ("ext forcing", []),
                ]
            ),
        ),
        (
            "simulation parameters",
            OrderedDict(
                [
                    ("version options", []),
                ]
            ),
        ),
    ]
)


def ncks_fl_fmt64bit(file_in, file_out=None):
    """
    Converts file to netCDF-3 64bit by calling:
      ncks --fl_fmt=64bit  file_in file_out

    Parameter
    ---------
    file : str
      The file to convert.
    """

    if file_out is None:
        file_out = file_in

    ncks_cmd = " ".join(["ncks", "-O", "--fl_fmt=64bit", file_in, file_out])
    cmd = " && ".join(["module load nco", ncks_cmd])

    p = Popen(cmd, stdout=PIPE, stderr=PIPE, shell=True)

    stdout, stderr = p.communicate()
    if p.returncode != 0:
        print(stdout.decode("UTF-8"))
        print(stderr.decode("UTF-8"))
        raise


def to_netcdf_clean(dset, path, format="NETCDF4", **kwargs):
    """wrap to_netcdf method to circumvent some xarray shortcomings"""

    import sys
    import subprocess
    from netCDF4 import default_fillvals

    dset = dset.copy()

    # ensure _FillValues are not added to coordinates
    for v in dset.coords:
        dset[v].encoding["_FillValue"] = None

    for v in dset.data_vars:
        if dset[v].dtype in [np.float32, np.float64]:
            dset[v].encoding["_FillValue"] = default_fillvals["f4"]
            dset[v].encoding["dtype"] = np.float32

        elif dset[v].dtype in [np.int32, np.int64]:
            dset[v].encoding["_FillValue"] = default_fillvals["i4"]
            dset[v].encoding["dtype"] = np.int32
        elif dset[v].dtype == object:
            pass
        else:
            warnings.warn(f"warning: unrecognized dtype for {v}: {dset[v].dtype}")

    sys.stderr.flush()

    print("-" * 30)
    print(f"Writing {path}")
    dset.to_netcdf(path, format=format, **kwargs)
    dumps = subprocess.check_output(["ncdump", "-h", path]).strip().decode("utf-8")
    print(dumps)
    dumps = subprocess.check_output(["ncdump", "-k", path]).strip().decode("utf-8")
    print(f"format: {dumps}")
    print("-" * 30)


def get_chem_mech_content(path):
    """read chem_mech.in file and return a dictionary
    with content divided into sections and subsections
    """
    with open(path) as fid:
        lines = fid.readlines()

    chem_mech = copy.deepcopy(chem_mech_template)

    N = 1
    key_section = list(chem_mech_template)[N]
    for i in range(len(lines)):
        value = lines[i].strip()
        if value.lower() == f"{key_section.lower()}":
            break
        chem_mech["header"].append(value)

    n = 0
    for l in lines[i + 1 :]:
        key_section = list(chem_mech_template)[N]
        key_subsection = list(chem_mech_template[key_section])[n]

        value = l.strip()
        if not value:
            continue

        if value.lower() == f"end {key_section.lower()}":
            N += 1
            n = 0

        elif value.lower() == f"end {key_subsection.lower()}":
            n = min(n + 1, len(list(chem_mech[key_section])) - 1)

        elif value.lower() not in [
            f"{key_section.lower()}",
            f"{key_subsection.lower()}",
        ]:
            chem_mech[key_section][key_subsection].append(value)

    return chem_mech


def write_chem_mech(chem_mech_dict, path):
    """write a chem_mech.in dictionary to file"""
    with open(path, "w") as fid:
        for section_key in chem_mech_dict.keys():
            if section_key == "header":
                for l in chem_mech_dict[section_key]:
                    fid.write(f"{l}\n")
            else:
                fid.write(f"{section_key.upper()}\n\n")
                for key in chem_mech_dict[section_key].keys():
                    fid.write(f"  {key.upper()}\n")

                    if chem_mech_dict[section_key][key]:
                        N = len(chem_mech_dict[section_key][key])
                        for n, l in enumerate(chem_mech_dict[section_key][key]):
                            val = l
                            if n == N - 1:
                                if val[-1] == ",":
                                    val = val[:-1]
                            fid.write(f"    {val}\n")
                    else:
                        fid.write(f"\n")

                    fid.write(f"  END {key.upper()}\n\n")
                fid.write(f"END {section_key.upper()}\n\n\n")


def cam_i_add_uniform_fields(ncdata_in, ncdata_out, tracer_dict, constant_ppm):
    ds = xr.open_dataset(ncdata_in, decode_times=False, decode_coords=False).load()

    for key, info in tracer_dict.items():
        assert "constituent" in info

        mw = molecular_weights[info["constituent"]]

        ic_constant = 1.0e-6 * constant_ppm * mw / molecular_weights["air"]
        var = xr.full_like(ds.CO2, fill_value=ic_constant)
        var.name = key
        var.attrs["long_name"] = key
        ds[key] = var

    to_netcdf_clean(ds, ncdata_out)
    ncks_fl_fmt64bit(ncdata_out)


def code_checkout(remote, coderoot, tag):
    """Checkout code for CESM
    If sandbox exists, check that the right tag has been checked-out.

    Otherwise, download the code, checkout the tag and run manage_externals.
    """

    sandbox = os.path.join(coderoot, tag)

    if os.path.exists(sandbox):
        print(f"Check for right tag: {sandbox}")
        p = Popen("git status", shell=True, cwd=sandbox, stdout=PIPE, stderr=PIPE)
        stdout, stderr = p.communicate()
        stdout = stdout.decode("UTF-8")
        stderr = stderr.decode("UTF-8")
        print(stdout)
        print(stderr)
        if tag not in stdout.split("\n")[0]:
            raise ValueError("tag does not match")

    else:
        if not os.path.exists(coderoot):
            os.makedirs(coderoot)

        # clone the repo
        p = Popen(
            f"git clone {remote} {tag}",
            shell=True,
            cwd=coderoot,
            stdout=PIPE,
            stderr=PIPE,
        )
        stdout, stderr = p.communicate()
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        if p.returncode != 0:
            raise Exception("git error")

        # check out the right tag
        assert os.path.exists(sandbox)
        p = Popen(f"git checkout {tag}", shell=True, cwd=sandbox)
        stdout, stderr = p.communicate()
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        if p.returncode != 0:
            raise Exception("git error")

        # check out externals
        p = Popen("./manage_externals/checkout_externals -v", shell=True, cwd=sandbox)
        stdout, stderr = p.communicate()
        if stdout:
            print(stdout)
        if stderr:
            print(stderr)
        if p.returncode != 0:
            raise Exception("git error")

    return sandbox
