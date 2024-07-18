# BSD 3-Clause License
#
# Copyright (c) 2018, Regents of the University of California
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""This module implements bag's interface with ngspice simulator.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Any, Sequence, Optional, List, Tuple, Union, Mapping, Set
from copy import copy

import re
import shutil
import time
from pathlib import Path
import numpy as np

from pybag.enum import DesignOutput
from pybag.core import get_cdba_name_bits

from ..math import float_to_si_string, si_string_to_float
from ..io.file import read_yaml, open_file, is_valid_file, read_file
from ..io.string import wrap_string
from ..util.immutable import ImmutableList
from .data import (
    MDSweepInfo, SimData, SetSweepInfo, SweepLinear, SweepLog, SweepList, SimNetlistInfo,
    SweepSpec, MonteCarlo, AnalysisInfo, AnalysisAC, AnalysisSP, AnalysisNoise, AnalysisTran,
    AnalysisSweep1D, AnalysisPSS, AnalysisPNoise, AnalysisDC
)
from .base import SimProcessManager, get_corner_temp
from .hdf5 import load_sim_data_hdf5, save_sim_data_hdf5
from .nutbin import NutBinParser

if TYPE_CHECKING:
    from .data import SweepInfo

reserve_params = {'freq', 'time'}


class NgspiceInterface(SimProcessManager):
    """This class handles interaction with Ngspice SPICE simulations.

    Supported Ngspice features:
        - Analyses: DC, AC, Tran, Noise, SP
        - Sources: All sources from Spectre, (v/i)trnoise
    Unsupported features: TODO
        - Features that require `control` loops and `alter`: in-sim sweep points (multiple corners,
            MC, param sweeps)
            - Control loops currently can't be used because they override the results file
                destination. The results file path currently can't be accessed by this SimAccess.
            - A workaround is to use BAG's GathererHelper to run multiple sweep points in parallel.
        - Analyses: DISTO, PZ, SENS, OP, TF
        - PSS (experimental in V42) and Pnoise (not supported by Pnoise)
        - XSPICE and CIDER

    Parameters
    ----------
    tmp_dir : str
        temporary file directory for SimAccess.
    sim_config : Dict[str, Any]
        the simulation configuration dictionary. Contains the following options:

        env_file : str
            the yaml path for PVT corners.
        compress : bool
            True to compress simulation data when saving to HDF5 file. Defaults to True.
        rtol: float
            relative tolerance for checking if 2 simulation values are the same. Defaults to 1e-8.
        atol: float
            absolute tolerance for checking if 2 simulation values are the same. Defaults to 1e-22.
        sim_options: Dict[str, Any]
            shared simulation options
        kwargs : Dict[str, Any]
            additional spectre simulation arguments. Contains the following options:

            command : str
                the command to launch simulator. Defaults to ngspice.
            env : Optional[Dict[str, str]]
                an optional dictionary of environment variables.  None to inherit from parent. Defaults to None.
            format : str
                the output raw data file format. Defaults to `nutbin` (nutmeg binary).
            options : List[str]
                the command line simulator options. Defaults to an empty list.
    """

    def __init__(self, tmp_dir: str, sim_config: Dict[str, Any]) -> None:
        SimProcessManager.__init__(self, tmp_dir, sim_config)
        self._model_setup: Dict[str, List[Tuple[str, str]]] = read_yaml(sim_config['env_file'])
        self._sim_kwargs: Dict[str, Any] = sim_config.get('kwargs', {})
        self._out_fmt: str = self._sim_kwargs.get('format', 'nutbin')
        self._sim_options: Dict[str, Any] = sim_config.get('sim_options', {})

    @property
    def netlist_type(self) -> DesignOutput:
        return DesignOutput.NGSPICE

    def create_netlist(self, output_path: Path, sch_netlist: Path, info: SimNetlistInfo,
                       precision: int = 6) -> None:
        output_path_str = str(output_path.resolve())
        sch_netlist_path_str = str(sch_netlist.resolve())
        if ('<' in output_path_str or '>' in output_path_str or
                '<' in sch_netlist_path_str or '>' in sch_netlist_path_str):
            raise ValueError('spectre does not support directory names with angle brackets.')

        sim_envs = info.sim_envs
        analyses = info.analyses
        params = info.params
        env_params = info.env_params
        swp_info = info.swp_info
        monte_carlo = info.monte_carlo
        sim_options = info.options
        init_voltages = info.init_voltages
        if monte_carlo is not None:
            raise NotImplementedError('Monte Carlo simulation not implemented for Ngspice')

        with open_file(sch_netlist, 'r') as f:
            lines = [l.rstrip() for l in f]

        # write simulator options
        _sim_options = copy(self._sim_options)
        _sim_options.update(sim_options)
        if _sim_options:
            sim_opt_list = ['.options']
            for opt, val in _sim_options.items():
                if not val:
                    sim_opt_list.append(f'{opt}')
                else:
                    sim_opt_list.append(f'{opt}={val}')
            sim_opt_str = wrap_string(sim_opt_list)
            lines.append(sim_opt_str)

        # write parameters
        param_fmt = '.param {}={}'
        param_set = reserve_params.copy()
        for par, val in swp_info.default_items():
            if par not in param_set:
                lines.append(param_fmt.format(par, _format_val(val, precision)))
                param_set.add(par)
        for par, val_list in env_params.items():
            if par in param_set:
                raise ValueError('Cannot set a sweep parameter as environment parameter.')
            lines.append(param_fmt.format(par, _format_val(val_list[0], precision)))
            param_set.add(par)
        for par, val in params.items():
            if par not in param_set:
                lines.append(param_fmt.format(par, _format_val(val, precision)))
                param_set.add(par)
        for ana in analyses:
            par = ana.param
            if par and par not in param_set:
                lines.append(param_fmt.format(par, _format_val(ana.param_start, precision)))
                param_set.add(par)

        lines.append('')

        # TODO support sweep. Requires using alter or altergroup, with a control statement
        if isinstance(swp_info, SetSweepInfo):
            # write paramset declaration if needed
            _write_param_set(lines, swp_info.params, swp_info.values, precision)
            lines.append('')

        if init_voltages:
            # write initial conditions
            ic_line = '.ic'
            for key, val in init_voltages.items():
                key = get_cdba_name_bits(key, DesignOutput.NGSPICE)
                if len(key) > 1:
                    raise ValueError("Separate initial bus into multiple values")
                key = key[0]
                ic_line += f' {key}={_format_val(val, precision)}'

            lines.append(ic_line)
            lines.append('')
            has_ic = True
        else:
            has_ic = False

        # write statements for each simulation environment
        # write default model statements
        # TODO Multiple sim environments, Monte carlo. Requires alter and a control statement
        for idx, sim_env in enumerate(sim_envs):
            if idx > 0:
                raise NotImplementedError("Multiple corners for a single sim in Ngspice not support yet.")

            corner, temp = get_corner_temp(sim_env)
            if idx != 0:
                # start altergroup statement
                lines.append(f'{sim_env} altergroup {{')
            _write_sim_env(lines, self._model_setup[corner], temp)
            if idx != 0:
                # write environment parameters for second sim_env and on
                for par, val_list in env_params.items():
                    lines.append(param_fmt.format(par, val_list[idx]))
                # close altergroup statement
                lines.append('}')
            lines.append('')

            # write sweep statements
            num_brackets = _write_sweep_start(lines, swp_info, idx, precision)
            if num_brackets > 0:
                raise NotImplementedError("Parameter sweeps in Ngspice not support yet.")

            # write analyses
            save_outputs = set()
            for ana in analyses:
                _write_analysis(lines, sim_env, ana, precision, has_ic)
                lines.append('')
                for output in ana.save_outputs:
                    try:
                        save_outputs.update(get_cdba_name_bits(output, DesignOutput.NGSPICE))
                    except ValueError:
                        save_outputs.update([output])


            # write save statements
            if isinstance(ana, AnalysisSP):
                # Writing a save statement for SP analysis prevents saving the S params. Skip.
                pass
            else:
                _write_save_statements(lines, save_outputs)
            
            # In Ngspice, we can't affect plotname without using a control statement
            # We can't write a control statement b/c we don't know the save file
            # Workaround to add title and corners to the top of the netlist, which becomes 
            #   the nutbin title. Then parse it in the NutbinParser
            # TODO: this fails if we have sweeps or alters
            # This syntax matches Spectre's, for compatibility
            lines_header = f"\"{ana.name} analysis `__{ana.name}__{sim_env}__'\""
            # Get inner sweep:
            if isinstance(ana, AnalysisAC):
                lines_header += f": freq = ({ana.sweep.start} -> {ana.sweep.stop})"
            elif isinstance(ana, AnalysisTran):
                lines_header += f": time = ({ana.start} -> {ana.stop})"
            elif isinstance(ana, AnalysisSP):
                lines_header += f": freq = ({ana.sweep.start} -> {ana.sweep.stop})"

            lines.insert(0, lines_header)

            # write end statement
            lines.append('.end')

        with open_file(output_path, 'w') as f:
            f.write('\n'.join(lines))
            f.write('\n')

    def get_sim_file(self, dir_path: Path, sim_tag: str) -> Path:
        return dir_path / f'{sim_tag}.hdf5'

    def load_sim_data(self, dir_path: Path, sim_tag: str) -> SimData:
        hdf5_path = self.get_sim_file(dir_path, sim_tag)
        import time
        print('Reading HDF5')
        start = time.time()
        ans = load_sim_data_hdf5(hdf5_path)
        stop = time.time()
        print(f'HDF5 read took {stop - start:.4g} seconds.')
        return ans

    async def async_run_simulation(self, netlist: Path, sim_tag: str) -> None:
        netlist = netlist.resolve()
        if not netlist.is_file():
            raise FileNotFoundError(f'netlist {netlist} is not a file.')

        sim_kwargs: Dict[str, Any] = self._sim_kwargs
        compress: bool = self.config.get('compress', True)
        rtol: float = self.config.get('rtol', 1e-8)
        atol: float = self.config.get('atol', 1e-22)

        cmd_str: str = sim_kwargs.get('command', 'ngspice')
        env: Optional[Dict[str, str]] = sim_kwargs.get('env', None)
        run_64: bool = sim_kwargs.get('run_64', True)  # TODO
        options = sim_kwargs.get('options', [])

        cwd_path = netlist.parent.resolve()
        log_path = cwd_path / 'ngspice_output.log'
        raw_path = cwd_path / 'sim.raw'
        hdf5_path: Path = cwd_path / f'{sim_tag}.hdf5'

        sim_cmd = [cmd_str, '-b', '-r', raw_path.name]

        for opt in options:
            sim_cmd.append(opt)

        sim_cmd.append(str(netlist))

        # delete previous .raw and .hdf5
        for fname in cwd_path.iterdir():
            if fname.name.startswith(raw_path.name) or fname.suffix == '.hdf5':
                try:
                    if fname.is_dir():
                        shutil.rmtree(str(fname))
                    elif fname.is_file():
                        fname.unlink()
                except FileNotFoundError:  # Ignore errors from race conditions
                    pass

        # Copy over .spiceinit
        spiceinit_file = self.config.get('spiceinit_file')
        if spiceinit_file:
            shutil.copy(Path(spiceinit_file), cwd_path)

        ret_code = await self.manager.async_new_subprocess(sim_cmd, str(log_path),
                                                           env=env, cwd=str(cwd_path))
        if ret_code is None or ret_code != 0:
            raise ValueError(f'Ngspice simulation ended with error.  See log file: {log_path}')

        # Check if raw_path is created. Give some slack for IO latency
        iter_cnt = 0
        while not (self._out_fmt.startswith('nut') and raw_path.is_file()):
            if iter_cnt > 120:
                raise ValueError(f'Ngspice simulation ended with error.  See log file: {log_path}')
            time.sleep(1)
            iter_cnt += 1

        # Check for log file existance 
        if not is_valid_file(log_path, '', 10, 1):
            raise ValueError(f'Ngspice simulation error: not log found. See log file: {log_path}')
        log_contents = read_file(log_path)
        
        # Choosing some possible error messages
        if 'Simulatin interrupted due to error' in log_contents:
            raise ValueError(f'Ngspice simulation ended with error. See log file: {log_path}')
        
        # Ngspice doesn't have a 'success' message. Choosing typical end message
        if 'ngspice program size' not in log_contents:
            raise ValueError(f'Ngspice simulation ended with error. See log file: {log_path}')

        # convert to HDF5
        if self._out_fmt == 'nutbin':
            nbp_mc = False
            for fname in cwd_path.iterdir():
                if str(fname).endswith('.mapping'):
                    nbp_mc = True
                    break
            nbp = NutBinParser(raw_path, rtol, atol, nbp_mc, parse_title=True, byte_order='<')
            save_sim_data_hdf5(nbp.sim_data, hdf5_path, compress)
            # post-process HDF5 to convert to MD array
            _process_hdf5(hdf5_path, rtol, atol)
        else:
            raise ValueError(f"Unsupported output type: {self._out_fmt}")


def _write_sim_env(lines: List[str], models: List[Tuple[str, str]], temp: int) -> None:
    for fname, section in models:
        if section:
            lines.append(f'.lib {fname} {section}')
        else:
            lines.append(f'.lib "{fname}"')
    lines.append(f'.temp={temp}')


def _write_param_set(lines: List[str], params: Sequence[str],
                     values: Sequence[ImmutableList[float]], precision: int) -> None:
    # get list of lists of strings to print, and compute column widths
    data = [params]
    col_widths = [len(par) for par in params]
    for combo in values:
        str_list = []
        for idx, val in enumerate(combo):
            cur_str = _format_val(val, precision)
            col_widths[idx] = max(col_widths[idx], len(cur_str))
            str_list.append(cur_str)
        data.append(str_list)

    # write the columns
    lines.append('swp_data paramset {')
    for row in data:
        lines.append(' '.join(val.ljust(width) for val, width in zip(row, col_widths)))
    lines.append('}')


def _get_sweep_str(par: str, swp_spec: Optional[SweepSpec], precision: int) -> str:
    # TODO
    # Sweeps in Ngspice require using loops and control statements. This currently break the
    #   write to file statements, since this interface does not know where to write the results.
    # Current solution is to rewrite the simulations to use GatherHelper and BAG sweeps.
    raise NotImplementedError("Support for Ngspice sweep currently not supported. See developer.")

    if not par or swp_spec is None:
        return ''

    if isinstance(swp_spec, SweepList):
        val_list = swp_spec.values
        # abstol check
        num_small = 0
        for val in val_list:
            if abs(val) < 3.0e-16:
                num_small += 1
        if num_small > 1:
            raise ValueError('sweep values are below spectre abstol, try to find a work around')

        tmp = ' '.join((_format_val(val, precision) for val in val_list))
        val_str = f'values=[{tmp}]'
    elif isinstance(swp_spec, SweepLinear):
        # spectre: stop is inclusive, lin = number of points excluding the last point
        val_str = f'start={swp_spec.start} stop={swp_spec.stop_inc} lin={swp_spec.num - 1}'
    elif isinstance(swp_spec, SweepLog):
        # spectre: stop is inclusive, log = number of points excluding the last point
        val_str = f'start={swp_spec.start} stop={swp_spec.stop_inc} log={swp_spec.num - 1}'
    else:
        raise ValueError('Unknown sweep specification.')

    if par in reserve_params:
        return val_str
    else:
        return f'param={par} {val_str}'


def _get_options_str(options: Mapping[str, str]) -> str:
    return ' '.join((f'{key}={val}' for key, val in options.items()))


def _write_sweep_start(lines: List[str], swp_info: SweepInfo, swp_idx: int, precision: int) -> int:
    #TODO 
    if isinstance(swp_info, MDSweepInfo):
        for dim_idx, (par, swp_spec) in enumerate(swp_info.params):
            statement = _get_sweep_str(par, swp_spec, precision)
            lines.append(f'swp{swp_idx}{dim_idx} sweep {statement} {{')
        return swp_info.ndim
    else:
        lines.append(f'swp{swp_idx} sweep paramset=swp_data {{')
        return 1


def _write_monte_carlo(lines: List[str], mc: MonteCarlo) -> int:
    #TOD
    cur_line = f'__{mc.name}__ montecarlo numruns={mc.numruns} seed={mc.seed}'
    options_dict = dict(savefamilyplots='yes', appendsd='yes', savedatainseparatedir='yes',
                        donominal='yes', variations='all')
    options_dict.update(mc.options)
    opt_str = _get_options_str(options_dict)
    if opt_str:
        cur_line += ' '
        cur_line += opt_str
    cur_line += ' {'
    lines.append(cur_line)
    return 1


def _write_analysis(lines: List[str], sim_env: str, ana: AnalysisInfo, precision: int,
                    has_ic: bool) -> List[str]:
    cur_line = f'.{ana.name}'

    if isinstance(ana, AnalysisTran):
        # Param order: tstep tstop <tstart <tmax>>
        # tstart indicates when the values start saving
        # for tstep, use Ngspice default of (tstop - tstart) / 50. Else strobe_period
        step_min = f"{{ ({ana.stop} - {ana.start}) / 50 }}"
        cur_line += (f' {step_min}' 
                     f' {_format_val(ana.stop, precision)}')

        if isinstance(ana.out_start, str) or ana.out_start > 0:
            cur_line += (f' {_format_val(ana.out_start, precision)}')
        else:
            cur_line += ' 0'

        if ana.strobe != 0:
            cur_line += (f' {_format_val(ana.strobe, precision)}')
        else:
            cur_line += (f' {step_min}')

        if has_ic:
            cur_line += ' uic'
    elif isinstance(ana, AnalysisDC):
        # Param order: srcnam, vstart, vstop, vincr
        par = ana.param  # TODO: caution sweeping src vs parameter
        vstart = ana.param_start
        vstop = ana.param_stop
        vincr = f'{{ ({vstop} - {vstart}) / {ana.sweep.num} }}'  # Assume linear sweep
        cur_line += f' {par} {{ {vstart} }}  {{ {vstop} }} {vincr}'
    elif isinstance(ana, AnalysisAC):
        # AnalysisSP and AnalysisNoise are subclasses of AnalysisAC
        # Param order for AC and SP : sweep type, num points, fstart, fstop
        # Param order for noise: portp, portn, src, sweep type, num points, fstart, fstop
        sweep = ana.sweep
        par = ana.param
        if par != 'freq':
            raise ValueError("AC sweeps in Ngspice other than freq currently not supported")
        sweep_type = sweep.type
        fstart = sweep.start
        fstop = sweep.stop
        if isinstance(sweep, SweepLinear):
            type_str = 'lin'
            num = sweep.num
        elif isinstance(sweep, SweepLog):
            # Ngspice doesn't directly support log. Instead, we will approximate using dec.
            type_str = 'dec'
            # TODO: what if fstop, fstart are parameter driven?
            _fstop = si_string_to_float(fstop) if isinstance(fstop, str) else fstop
            _fstart = si_string_to_float(fstart) if isinstance(fstart, str) else fstart
            num = np.ceil(sweep.num / (np.log10(_fstop) - np.log10(_fstart)))
        else:
            raise RuntimeError("Unsupported sweep type: ", sweep_type)
        # Additional legwork for noise
        if isinstance(ana, AnalysisNoise):
            # noise analysis also requires the output port (and reference) and an
            #   an input referred probe. The input referred probe may not be given, in which case
            #   the output probe will be used.

            # Get probed (output) node
            if ana.n_port:
                if not ana.p_port:
                    raise ValueError("If p_port is specified, n_port must also be specified!")
                probe_str = f'v({ana.p_port}, {ana.n_port})'
            elif ana.p_port:
                # Assume ref is VSS
                probe_str = f'v({ana.p_port})'
            elif ana.out_probe:
                print("WARNING: p_port not specified, but out_probe is. Assuming out_probe is connected to net 'out_probe'...")
                if ana.out_probe.startswith('V') or ana.out_probe.startswith('v'):
                    raise ValueError("Ngspice cannot use voltage source (current probe) for out_probe.")
                probe_str = f'v({ana.out_probe})'
            else:
                raise ValueError('Either specify out_probe, or specify p_port and n_port, or specify measurement.')
            cur_line += f' {probe_str.lower()}'

            # Get input referred source
            # TODO: can we insert a dummy somewhere?
            if ana.in_probe:
                cur_line += f' {ana.in_probe}'
            else:
                raise ValueError("Ngspice requires a voltage source for in_probe.")

        cur_line += f' {type_str} {num} {fstart} {fstop}'
    elif isinstance(ana, AnalysisPSS):
        # As of Ngspice V42, PSS is still experimental and not publicly available.
        # Marking as not implemented
        raise NotImplementedError("Ngspice does not support PSS!")
        # Order: gfreq, tstab, oscnob, psspoints, harms, sciter, steadycoeff, <uic>
        if ana.period == 0.0 and ana.fund == 0.0 and ana.autofund is False:
            raise ValueError('For PSS simulation, either specify period or fund, '
                             'or set autofund = True')
        if ana.period > 0.0:
            gfreq = f'{1/ana.period}'
        elif ana.fund > 0.0:
            gfreq = f'{ana.fund}'
        else:
            raise ValueError("PSS: no guess period provided")
        cur_line += f' {gfreq}'

        cur_line += f" {ana.options.get('tstab', f'{{20/{gfreq}}}')}"
        cur_line += f" {ana.p_port}"
        cur_line += f" {ana.options.get('pss_points', 1024)}"
        cur_line += f" {ana.options.get('num_harms', 7)}"
        cur_line += f" {ana.options.get('sciter', 50)}"
        cur_line += f" {ana.options.get('steady_coeff', 1e-3)}"
    
        if has_ic:
            cur_line += ' uic'

        if ana.autofund:
            raise ValueError("Ngspice does not have support for autofund")
        if ana.strobe != 0:
            raise ValueError("Ngspice does not have support for strobe period")
    else:
        raise ValueError('Unknown analysis specification.')

    lines.append(cur_line)

    jitter_event = []
    if isinstance(ana, AnalysisPNoise):
        raise ValueError("Ngspice does not support Pnoise!")
    return jitter_event


def _write_save_statements(lines: List[str], save_outputs: Set[str]) -> None:
    if not save_outputs:
        return

    out_str = '.save'
    probes: List[str] = []
    for save_out in sorted(save_outputs):
        if ':' in save_out:
            probes.append(save_out)
        else:
            out_str += f' {save_out}'

    # Convert probe statements
    curr_lines = []
    for probe in probes:
        inst, term = probe.split(':')
        if term == 'pwr':
            curr_lines.append(f'.probe p({inst})')
        else:
            curr_lines.append(f'.probe I({inst},{term})')

    for line in curr_lines:
        lines.append(line)

    lines.append(out_str)


def _format_val(val: Union[float, str], precision: int = 6) -> str:
    if isinstance(val, str):
        return f"{{{val}}}"
    else:
        return float_to_si_string(val, precision)


def _process_hdf5(path: Path, rtol: float, atol: float) -> None:
    proc = 'process'
    sim_data = load_sim_data_hdf5(path)
    modified = False
    for grp in sim_data.group_list:
        sim_data.open_group(grp)
        if proc in sim_data.sweep_params:
            modified |= sim_data.remove_sweep(proc, rtol=rtol, atol=atol)

    if modified:
        save_sim_data_hdf5(sim_data, path)
