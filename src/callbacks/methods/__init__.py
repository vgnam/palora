from .algo_callback import AlgoCallback, ParetoFrontApproximationAlgoCallback

#
from .auto_lambda import AutoLambda
from .cagrad import CAGrad
from .cosmos import COSMOS
from .dwa import DynamicWeightAverage
from .graddrop import GradDrop
from .imtl import IMTLG
from .ls import LinearScalarization
from .mgda import MGDA
from .nashmtl import NashMTL
from .palora import PaLoRA, PaLoRA_GB, PaLoRA_LB, PaLoRAFull
from .palora_qd import PaLoRAQD

#
from .pamal import PaMaL, PaMaL_GB, PaMaL_LB
from .pamal_mc_div import PaMaLMCDiv
from .pamal_qd import PaMaLQD
from .pagel import PaGeL
from .pcgrad import PCGrad
from .rlw import RandomLossWeighting
from .si import ScaleInvariantLinearScalarization
from .stl import STL
from .uw import UncertaintyWeighting
from .phn import ParetoHyperNetwork

from .padora import PaDoRA, PaDoRA_GB, PaDoRA_LB, PaDoRAFull

METHODS = dict(
    auto_lambda=AutoLambda,
    autol=AutoLambda,
    cagrad=CAGrad,
    dwa=DynamicWeightAverage,
    graddrop=GradDrop,
    imtl=IMTLG,
    ls=LinearScalarization,
    mgda=MGDA,
    nashmtl=NashMTL,
    pcgrad=PCGrad,
    rlw=RandomLossWeighting,
    si=ScaleInvariantLinearScalarization,
    stl=STL,
    uw=UncertaintyWeighting,
    #
    # Pareto Front Approximation
    #
    pamal=PaMaL,
    pamal_lb=PaMaL_LB,
    pamal_gb=PaMaL_GB,
    pamal_mc_div=PaMaLMCDiv,
    pamal_qd=PaMaLQD,
    pagel=PaGeL,
    palora=PaLoRA,
    palora_lb=PaLoRA_LB,
    palora_gb=PaLoRA_GB,
    palora_full=PaLoRAFull,
    palora_qd=PaLoRAQD,
    cosmos=COSMOS,
    phn=ParetoHyperNetwork,
    #
    padora=PaDoRA,
    padora_gb=PaDoRA_GB,
    padora_lb=PaDoRA_LB,
    padora_full=PaDoRAFull,
)

PFL_METHODS = dict(
    pamal=PaMaL,
    pamal_lb=PaMaL_LB,
    pamal_gb=PaMaL_GB,
    pamal_mc_div=PaMaLMCDiv,
    pamal_qd=PaMaLQD,
    pagel=PaGeL,
    palora=PaLoRA,
    palora_lb=PaLoRA_LB,
    palora_gb=PaLoRA_GB,
    palora_full=PaLoRAFull,
    palora_qd=PaLoRAQD,
    cosmos=COSMOS,
    phn=ParetoHyperNetwork,
    #
    padora=PaDoRA,
    padora_gb=PaDoRA_GB,
    padora_lb=PaDoRA_LB,
    padora_full=PaDoRAFull,
)


def get_method(name, num_tasks, **kwargs) -> AlgoCallback:
    if name == "phn":
        return ParetoHyperNetwork(num_tasks=num_tasks, **kwargs)

    if "palora" not in name and "pamal" not in name and "padora" not in name:
        return METHODS[name](num_tasks, **kwargs)

    if name == "palora_qd":
        return PaLoRAQD(num_tasks=num_tasks, **kwargs)

    if "palora" in name:
        inner_method = kwargs.get("inner_method")
        if inner_method == "ls":
            return PaLoRA(num_tasks, **kwargs)
        elif inner_method == "gb":
            return PaLoRA_GB(num_tasks, **kwargs)
        elif inner_method == "lb":
            return PaLoRA_LB(num_tasks, **kwargs)
        elif inner_method == "full":
            return PaLoRAFull(num_tasks, **kwargs)
        else:
            raise ValueError(f"Unknown inner method {inner_method}")

    if name == "pamal_mc_div":
        return PaMaLMCDiv(num_tasks, **kwargs)

    if name == "pagel":
        return PaGeL(num_tasks=num_tasks, **kwargs)

    if name == "pamal_qd":
        return PaMaLQD(num_tasks=num_tasks, **kwargs)

    if "pamal" in name:
        inner_method = kwargs.get("inner_method")
        if inner_method == "ls":
            return PaMaL(num_tasks, **kwargs)
        elif inner_method == "gb":
            return PaMaL_GB(num_tasks, **kwargs)
        elif inner_method == "lb":
            return PaMaL_LB(num_tasks, **kwargs)
        else:
            raise ValueError(f"Unknown inner method {inner_method}")

    if "padora" in name:
        inner_method = kwargs.get("inner_method")
        if inner_method == "ls":
            return PaDoRA(num_tasks, **kwargs)
        elif inner_method == "gb":
            return PaDoRA_GB(num_tasks, **kwargs)
        elif inner_method == "lb":
            return PaDoRA_LB(num_tasks, **kwargs)
        elif inner_method == "full":
            return PaDoRAFull(num_tasks, **kwargs)
        else:
            raise ValueError(f"Unknown inner method {inner_method}")

    raise ValueError(f"Unknown method {name}")


__all__ = [
    "METHODS",
    "PFL_METHODS",
    "AlgoCallback",
    "ParetoFrontApproximationAlgoCallback",
    "AutoLambda",
    "CAGrad",
    "DynamicWeightAverage",
    "GradDrop",
    "IMTLG",
    "LinearScalarization",
    "MGDA",
    "NashMTL",
    "PCGrad",
    "RandomLossWeighting",
    "ScaleInvariantLinearScalarization",
    "STL",
    "UncertaintyWeighting",
    "PaMaL",
    "PaMaL_LB",
    "PaMaL_GB",
    "PaLoRA",
    "PaLora_LB",
    "PaLoRA_GB",
    "PaLoRAFull",
    "COSMOS",
    "get_method",
]
