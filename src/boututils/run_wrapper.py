"""Collection of functions which can be used to make a BOUT++ run"""

import os
import pathlib
import re
import subprocess
from subprocess import PIPE, STDOUT, Popen, call

if os.name == "nt":
    # Default on Windows
    DEFAULT_MPIRUN = "mpiexec.exe -n"
else:
    DEFAULT_MPIRUN = "mpirun -np"


def getmpirun(default=DEFAULT_MPIRUN):
    """Return environment variable named MPIRUN, if it exists else return
     a default mpirun command

    Parameters
    ----------
    default : str, optional
        An mpirun command to return if ``MPIRUN`` is not set in the environment

    """
    MPIRUN = os.getenv("MPIRUN")

    if MPIRUN is None or MPIRUN == "":
        MPIRUN = default
        print(f"getmpirun: using the default {default}")

    return MPIRUN


def shell(command, pipe=False, env=None):
    """Run a shell command

    Parameters
    ----------
    command : list of str
        The command to run, split into (shell) words
    pipe : bool, optional
        Grab the output as text, else just run the command in the
        background

    Returns
    -------
    tuple : (int, str)
        The return code, and either command output if pipe=True else None
    """

    result = subprocess.run(
        command, shell=True, capture_output=pipe, env=env, text=True
    )
    output = result.stdout if pipe else ""
    if result.stderr:
        output += "\nSTDERR:\n" + result.stderr
    return result.returncode, output


def determineNumberOfCPUs():
    """Number of virtual or physical CPUs on this system

    i.e. user/real as output by time(1) when called with an optimally
    scaling userspace-only program

    Taken from a post on stackoverflow:
    https://stackoverflow.com/questions/1006289/how-to-find-out-the-number-of-cpus-in-python

    Returns
    -------
    int
        The number of CPUs
    """

    # cpuset
    # cpuset may restrict the number of *available* processors
    try:
        m = re.search(r"(?m)^Cpus_allowed:\s*(.*)$", open("/proc/self/status").read())
        if m:
            res = bin(int(m.group(1).replace(",", ""), 16)).count("1")
            if res > 0:
                return res
    except OSError:
        pass

    # Python 2.6+
    try:
        import multiprocessing

        return multiprocessing.cpu_count()
    except (ImportError, NotImplementedError):
        pass

    # POSIX
    try:
        res = int(os.sysconf("SC_NPROCESSORS_ONLN"))

        if res > 0:
            return res
    except (AttributeError, ValueError):
        pass

    # Windows
    try:
        res = int(os.environ["NUMBER_OF_PROCESSORS"])

        if res > 0:
            return res
    except (KeyError, ValueError):
        pass

    # jython
    try:
        from java.lang import Runtime

        runtime = Runtime.getRuntime()
        res = runtime.availableProcessors()
        if res > 0:
            return res
    except ImportError:
        pass

    # BSD
    try:
        sysctl = subprocess.Popen(["sysctl", "-n", "hw.ncpu"], stdout=subprocess.PIPE)
        scStdout = sysctl.communicate()[0]
        res = int(scStdout)

        if res > 0:
            return res
    except (OSError, ValueError):
        pass

    # Linux
    try:
        res = open("/proc/cpuinfo").read().count("processor\t:")

        if res > 0:
            return res
    except OSError:
        pass

    # Solaris
    try:
        pseudoDevices = os.listdir("/devices/pseudo/")
        expr = re.compile("^cpuid@[0-9]+$")

        res = 0
        for pd in pseudoDevices:
            if expr.match(pd) is not None:
                res += 1

        if res > 0:
            return res
    except OSError:
        pass

    # Other UNIXes (heuristic)
    try:
        try:
            dmesg = open("/var/run/dmesg.boot").read()
        except OSError:
            dmesgProcess = subprocess.Popen(["dmesg"], stdout=subprocess.PIPE)
            dmesg = dmesgProcess.communicate()[0]

        res = 0
        while f"\ncpu{res}:" in dmesg:
            res += 1

        if res > 0:
            return res
    except OSError:
        pass

    raise Exception("Can not determine number of CPUs on this system")


def launch(
    command,
    runcmd=None,
    nproc=None,
    mthread=None,
    output=None,
    pipe=False,
    verbose=False,
):
    """Launch parallel MPI jobs

    >>> status = launch(command, nproc, output=None)

    Parameters
    ----------
    command : str
        The command to run
    runcmd : str, optional
        Command for running parallel job; defaults to what getmpirun() returns"
    nproc : int, optional
        Number of processors (default: all available processors)
    mthread : int, optional
        Number of omp threads (default: the value of the
        ``OMP_NUM_THREADS`` environment variable
    output : str, optional
        Name of file to save output to
    pipe : bool, optional
        If True, return the output of the command
    verbose : bool, optional
        Print the full command to be run before running it

    Returns
    -------
    tuple : (int, str)
        The return code, and either command output if pipe=True else None

    """

    if runcmd is None:
        runcmd = getmpirun()

    if nproc is None:
        # Determine number of CPUs on this machine
        nproc = determineNumberOfCPUs()

    cmd = f"{runcmd} {nproc} {command}"

    if output is not None:
        cmd = f"{cmd} > {output}"

    # Set OMP_NUM_THREADS if mthread is provided (for OpenMP in BOUT++)
    env = dict(os.environ)
    if mthread:
        env["OMP_NUM_THREADS"] = str(mthread)

    if verbose:
        print(cmd)

    return shell(cmd, pipe=pipe, env=env)


def shell_safe(command, *args, **kwargs):
    """'Safe' version of shell.

    Raises a `RuntimeError` exception if the command is not
    successful

    Parameters
    ----------
    command : str
        The command to run
    *args, **kwargs
        Optional arguments passed to `shell`

    """
    s, out = shell(command, *args, **kwargs)
    if s:
        raise RuntimeError(
            f"Run failed with {s}.\nCommand was:\n{command}\n\nOutput was\n\n{out}"
        )
    return s, out


def launch_safe(command, *args, **kwargs):
    """'Safe' version of launch.

    Raises an RuntimeError exception if the command is not successful

    Parameters
    ----------
    command : str
        The command to run
    *args, **kwargs
        Optional arguments passed to `shell`

    """

    s, out = launch(command, *args, **kwargs)
    if s != 0:
        raise RuntimeError(
            f"Run failed with {s}.\nCommand was:\n{command}\n\nOutput was\n\n{out}"
        )
    return s, out


def build_and_log(test):
    """Run make and redirect the output to a log file. Prints input

    On Windows, does nothing because executable should have already
    been built

    """

    if os.name == "nt":
        return

    print(f"Making {test}")

    if os.path.exists("makefile") or os.path.exists("Makefile"):
        return shell_safe("make > make.log")

    ctest_filename = "CTestTestfile.cmake"
    if not os.path.exists(ctest_filename):
        raise RuntimeError("Could not build: no makefile and no CMake files detected")

    # We're using CMake, but we need to know the target name. If
    # bout_add_integrated_test was used (which it should have been!),
    # then the test name is the same as the target name
    with open(ctest_filename) as f:
        contents = f.read()
    match = re.search("add_test.(.*) ", contents)
    if match is None:
        raise RuntimeError("Using CMake, but could not determine test name")
    test_name = match.group(1).split()[0]

    # Now we need to find the build directory. It'll be the first
    # parent containing CMakeCache.txt
    here = pathlib.Path(".").absolute()
    for parent in here.parents:
        if (parent / "CMakeCache.txt").exists():
            return shell_safe(f"cmake --build {parent} --target {test_name} > make.log")

    # We've just looked up the entire directory structure and not
    # found the build directory, this could happen if CMakeCache was
    # deleted, in which case we can't build anyway
    raise RuntimeError("Using CMake, but could not find build directory")
