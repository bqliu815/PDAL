using Printf
import LinearAlgebra
import CUDA

_cuda_norm_shim(args...) = LinearAlgebra.norm(args...)
_cuda_dot_shim(args...) = LinearAlgebra.dot(args...)
if !isdefined(CUDA, :norm)
    @eval CUDA norm(args...) = Main._cuda_norm_shim(args...)
end
if !isdefined(CUDA, :dot)
    @eval CUDA dot(args...) = Main._cuda_dot_shim(args...)
end

using HPRLP

function getarg(i::Int, default=nothing)
    return length(ARGS) >= i ? ARGS[i] : default
end

instance = getarg(1)
out_csv = getarg(2)
tol = parse(Float64, getarg(3, "1e-8"))
time_limit = parse(Float64, getarg(4, "3600"))

if instance === nothing || out_csv === nothing
    error("usage: julia --project run_hprlp_one.jl INSTANCE OUT_CSV TOL TIME_LIMIT")
end

mkpath(dirname(out_csv))
name = replace(basename(instance), r"\.mps$"i => "")
split = basename(dirname(instance))
t_wall = time()

open(out_csv, "w") do io
    println(io, "solver,split,instance,status,time_sec,iterations,objective,rel_gap,rel_primal,rel_dual,error")
    try
        params = HPRLP.HPRLP_parameters()
        params.stoptol = tol
        params.time_limit = time_limit
        params.use_gpu = true
        params.device_number = 0
        params.warm_up = true
        result = HPRLP.run_single(instance, params)
        # HPR-LP exposes one combined KKT stopping residual. Repeat it in the
        # common primal/dual columns so that their maximum remains exact.
        @printf(io, "HPR-LP(Julia),%s,%s,%s,%.12g,%d,%.12g,%.12g,%.12g,%.12g,\n",
            split,
            name,
            result.output_type,
            result.time,
            result.iter,
            result.primal_obj,
            result.gap,
            result.residuals,
            result.residuals,
        )
    catch err
        msg = replace(sprint(showerror, err), '\n' => ' ')
        @printf(io, "HPR-LP(Julia),%s,%s,ERROR,%.12g,0,NaN,NaN,NaN,NaN,%s\n",
            split,
            name,
            time() - t_wall,
            replace(msg, ',' => ';'),
        )
    end
end
