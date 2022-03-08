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
XiO2 = 0.2095  # Mean atmospheric O2 mixing ratio

molecules_per_mol = 6.0221413e+23
m2_per_cm2 = 1e-4

radius_earth = 6.37122e6  # m

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

        # if dtype has been set explicitly in encoding, obey
        if "dtype" in dset[v].encoding:
            if dset[v].encoding["dtype"] == np.float64:
                dset[v].encoding["_FillValue"] = default_fillvals["f8"]
            elif dset[v].encoding["dtype"] == np.float32:
                dset[v].encoding["_FillValue"] = default_fillvals["f4"]
            elif dset[v].encoding["dtype"] in [np.int32]:
                dset[v].encoding["_FillValue"] = default_fillvals["i4"]

        # otherwise, default to single precision output
        elif dset[v].dtype in [np.float32, np.float64]:
            dset[v].encoding["_FillValue"] = default_fillvals["f4"]
            dset[v].encoding["dtype"] = np.float32

        elif dset[v].dtype in [np.int32, np.int64]:
            dset[v].encoding["_FillValue"] = default_fillvals["i4"]
            dset[v].encoding["dtype"] = np.int32
        elif dset[v].dtype == object:
            pass
        else:
            warnings.warn(f"warning: unrecognized dtype for {v}: {dset[v].dtype}")

        if "_FillValue" not in dset[v].encoding:
            dset[v].encoding["_FillValue"] = None

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


def cam_i_add_uniform_fields(ncdata_in, ncdata_out, tracer_dict, background_ppm):
    ds = xr.open_dataset(ncdata_in, decode_times=False, decode_coords=False).load()

    for key, info in tracer_dict.items():
        assert "constituent" in info

        mw = molecular_weights[info["constituent"]]

        ic_constant = 1.0e-6 * background_ppm * mw / molecular_weights["air"]
        var = xr.full_like(ds.Q, fill_value=ic_constant)
        var.name = key
        var.attrs["long_name"] = key
        ds[key] = var

    ds.attrs["cam_i_add_uniform_fields_background_ppm"] = 400.0
    to_netcdf_clean(ds, ncdata_out)
    ncks_fl_fmt64bit(ncdata_out)


def fincl_lonlat_to_dataset(ds, specifer_dict, isel_dict={}):
    """
    Convert a dataset written by CAM using the fincNlonlat mechanism to
    a more usable format, replacing each of the individual variables with
    a `record` dimension.

    This converts variables that might look like this::

        float32 PS_62.507w_82.451n(time, lat_82.451n, lon_62.507w) ;
                    PS_62.507w_82.451n:units = Pa ;
                    PS_62.507w_82.451n:long_name = Surface pressure ;
                    PS_62.507w_82.451n:cell_methods = time: mean ;
                    PS_62.507w_82.451n:basename = PS ;
        float32 PS_156.611w_71.323n(time, lat_71.323n, lon_156.611w) ;
                    PS_156.611w_71.323n:units = Pa ;
                    PS_156.611w_71.323n:long_name = Surface pressure ;
                    PS_156.611w_71.323n:cell_methods = time: mean ;
                    PS_156.611w_71.323n:basename = PS ;

    To this::

        float32 PS(record, time) ;
            PS:units = Pa ;
            PS:long_name = Surface pressure ;
            PS:cell_methods = time: mean ;
            PS:basename = PS ;

    Where the `record` dimension is determined from the `specifer_dict`, which might
    look like this::

        specifer_dict = {
             'alt': '62.507w_82.451n',
             'brw': '156.611w_71.323n',
             'cba': '162.720w_55.210n',
             'cgo': '144.690e_40.683s',
             'gould_57S': '64.222w_57.023s',
             'gould_59S': '63.317w_59.026s',
             'gould_61S': '60.621w_61.042s',
             'gould_63S': '61.123w_63.077s',
             'gould_65S': '63.855w_64.785s',
             'kum': '154.888w_19.561n',
             'ljo': '117.257w_32.867n',
             'mlo': '155.576w_19.536n',
             'psa': '64.053w_64.774s',
             'smo': '170.564w_14.247s',
             'spo': '24.800w_89.980s',
        }

    I.e., the keys in the dictionary are used to generate, `record`, which is a
    new coordinate variable::

        xarray.DataArray 'record' record: 15
            array(['alt', 'brw', 'cba', 'cgo', 'gould_57S', 'gould_59S', 'gould_61S',
                   'gould_63S', 'gould_65S', 'kum', 'ljo', 'mlo', 'psa', 'smo', 'spo'],
                  dtype='<U9')
            Coordinates:
              record (record) <U9 'alt' 'brw' 'cba' ... 'smo' 'spo'

    """
    keys = list(specifer_dict.keys())
    lonlat = specifer_dict[keys[0]]
    data_vars = {v.replace(f"_{lonlat}", "") for v in ds.data_vars if lonlat in v}

    record = xr.DataArray(keys, dims=("record"), name="record")

    dso = ds[[c for c in ds.coords if "lat_" not in c and "lon_" not in c]]
    for v in data_vars:
        da_list = []
        for key in keys:
            lonlat = specifer_dict[key]
            dims = (f"lat_{lonlat.split('_')[1]}", f"lon_{lonlat.split('_')[0]}")
            da = ds[f"{v}_{lonlat}"].isel({dims[0]: 0, dims[1]: 0}, drop=True)
            da.name = v
            assert (
                ds[f"{v}_{lonlat}"].dims[-2:] == dims
            ), f"expecting: {dims}\ngot: {ds[f'{v}_{lonlat}'].dims}"
            da_list.append(da)
        dso[v] = xr.concat(da_list, dim=record)

    if isel_dict:
        dso = dso.isel(**isel_dict)
    return dso


def tracegas_convert_units(da, constituent, background_ppm):
    """
    Convert the units of trace gas constituents that have been
    simulated using an initial background concentration.

    See also `cam_i_add_uniform_fields`.

    """
    units = da.attrs["units"]
    assert units in ["kg/kg", "mol/mol"], f"unknown units: {units}\n{da}"

    if constituent == "O2":
        units_out = "per meg"
    elif constituent in ["CO2", "N2"]:
        units_out = "ppm"
    else:
        raise ValueError(f"unknown constituent: {constituent}")

    mwair = molecular_weights["air"]
    mw = molecular_weights[constituent]

    with xr.set_options(keep_attrs=True):
        if units == "kg/kg":
            da_converted = da * 1.0e6 * mwair / mw - background_ppm
        elif units == "mol/mol":
            da_converted = da * 1.0e6 - background_ppm
        if constituent == "O2":
            da_converted /= XiO2

    da_converted.attrs["units"] = units_out
    return da_converted


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


def dim_cnt_check(ds, varname, dim_cnt):
    """confirm that varname in ds has dim_cnt dimensions"""
    if len(ds[varname].dims) != dim_cnt:
        raise ValueError(
            f"unexpected dim_cnt={len(ds[varname].dims)}, varname={varname}"
        )


def get_area(ds, component):
    """return area DataArray appropriate for component"""
    if component == "ocn":
        dim_cnt_check(ds, "TAREA", 2)
        return ds["TAREA"]
    if component == "ice":
        dim_cnt_check(ds, "tarea", 2)
        return ds["tarea"]
    if component == "lnd":
        dim_cnt_check(ds, "landfrac", 2)
        dim_cnt_check(ds, "area", 2)
        da_ret = ds["landfrac"] * ds["area"]
        da_ret.name = "area"
        da_ret.attrs["units"] = ds["area"].attrs["units"]
        return da_ret
    if component == "atm":
        dim_cnt_check(ds, "gw", 1)
        area_earth = 4.0 * np.pi * radius_earth ** 2  # area of earth in CIME [m2]

        # normalize area so that sum over "lat", "lon" yields area_earth
        area = ds["gw"] + 0.0 * ds["lon"]  # add "lon" dimension
        area = (area_earth / area.sum(dim=("lat", "lon"))) * area
        area.attrs["units"] = "m2"
        return area
    msg = f"unknown component={component}"
    raise ValueError(msg)
