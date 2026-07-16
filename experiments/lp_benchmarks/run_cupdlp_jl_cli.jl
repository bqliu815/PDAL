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

include(joinpath(ENV["ROOT"], "src", "cuPDLP-jl-master", "scripts", "solve.jl"))
