cd(@__DIR__)
using LinearAlgebra, Plots, Random

case = 2 # case 1: noise-free measurement, case 2: noisy measurement

## data generation
dT = 0.05
t = 0:dT:5
θ = ones(4)
Φ(s) = [sin(s) cos(2.0 * s) sin(0.5 * s^2 + s) cos(0.5 * s^2 + s)]
f(s) = Φ(s) * θ
y_true = vcat(f.(t)...)
rng = MersenneTwister(1234)
noise = 0.5 * randn(rng, length(t))

if case == 1
    y = y_true
elseif case == 2
    y = y_true + noise
end

# f_y = plot(t, y, xlabel = "\$t\$ [s]", ylabel = "\$y\$", label = "\$y_{\\textrm{true}}\$", legend = :topright)
# display(f_y)

## update rules
function RLS(θ̂_prev, R_prev, y_current, Φ_current)
    K = R_prev * Φ_current' / (I + Φ_current * R_prev * Φ_current')
    θ̂_current = θ̂_prev + K * (y_current - Φ_current * θ̂_prev)
    R_current = R_prev - K * Φ_current * R_prev
    return (θ̂_current, R_current)
end

function ORFit(θ̂_prev, P_prev, y_current, Φ_current)
    K = P_prev * pinv(Φ_current * P_prev)
    θ̂_current = θ̂_prev + K * (y_current - Φ_current * θ̂_prev)
    P_current = P_prev - K * Φ_current * P_prev
    return (θ̂_current, P_current)
end

function inf_accum(Ω_prev, Φ_current)
    Ω_current = Ω_prev + Φ_current' * Φ_current
    return Ω_current
end

## Standard RLS
ρ_list = [1.0, 1E1, 1E2, 1E3]
dat_RLS = Vector{Any}(undef, length(ρ_list))

for i in eachindex(ρ_list)
    θ̂_prev = zeros(4)
    R_prev = ρ_list[i] * I(4)

    θ̂_dat_RLS = Vector{Any}(undef, length(t))
    # R_dat = Vector{Any}(undef,length(t))
    ŷ_dat_RLS = Vector{Any}(undef, length(t))

    for k in eachindex(t)
        (θ̂_current, R_current) = RLS(θ̂_prev, R_prev, f(t[k]), Φ(t[k]))
        θ̂_dat_RLS[k] = θ̂_current
        # R_dat[k] = R_current
        ŷ_dat_RLS[k] = Φ(t[k]) * θ̂_current
        # global (θ̂_prev, R_prev) = (θ̂_current, R_current)
        (θ̂_prev, R_prev) = (θ̂_current, R_current)
    end

    ŷ_RLS = vcat(ŷ_dat_RLS...)
    e_y_RLS = [abs(y_true[k] - ŷ_dat_RLS[k][]) for k in 1:length(t)]
    # f_y_RLS = plot(t, [y ŷ_RLS], xlabel = "\$t\$ [s]", ylabel = "\$y\$", label = ["\$y_{\\textrm{true}}\$" "\$\\hat{y}\$"], legend = :topright)
    # display(f_y_RLS)

    e_θ_RLS = [norm(θ - θ̂_dat_RLS[k]) for k in 1:length(t)]
    # f_e_θ_RLS = plot(t, e_θ_RLS, xlabel = "\$t\$ [s]", ylabel = "\$ e_{\\theta} = \\left| \\hat{\\theta} - \\theta\\right|\$", label = "RLS", legend = :topright)
    # display(f_e_θ_RLS)

    dat_RLS[i] = (; θ̂_dat_RLS = θ̂_dat_RLS, ŷ_RLS = ŷ_RLS, e_y_RLS = e_y_RLS, e_θ_RLS = e_θ_RLS)
end

## Two-Stage ORFit+RLS
θ̂_prev = zeros(4)
P_prev = I(4)
Ω_prev = zeros(4, 4)
mode = 1

θ̂_dat_2RLS = Vector{Any}(undef, length(t))
# P_dat = Vector{Any}(undef,length(t))
ŷ_dat_2RLS = Vector{Any}(undef, length(t))

for k in eachindex(t)
    global mode
    if mode == 1
        Ω_current = inf_accum(Ω_prev, Φ(t[k]))
        if rank(Ω_current) == 4
            mode = 2
        end
        if ~isequal(mode, 2)
            (θ̂_current, P_current) = ORFit(θ̂_prev, P_prev, f(t[k]), Φ(t[k]))
        end
        global P_prev = P_current
    end

    if mode == 2
        mode = 3
        global R_prev = inv(Ω_current)
        θ̂_current = θ̂_prev + Ω_current \ Φ(t[k])' * (f(t[k]) - Φ(t[k]) * θ̂_prev)

    elseif mode == 3
        (θ̂_current, R_current) = RLS(θ̂_prev, R_prev, f(t[k]), Φ(t[k]))
        global R_prev = R_current
    end

    θ̂_dat_2RLS[k] = θ̂_current
    ŷ_dat_2RLS[k] = Φ(t[k]) * θ̂_current
    global θ̂_prev = θ̂_current
end

ŷ_2RLS = vcat(ŷ_dat_2RLS...)
e_y_2RLS = [abs(y_true[k] - ŷ_dat_2RLS[k][]) for k in 1:length(t)]
# f_y_2RLS = plot(t, [y ŷ_2RLS], xlabel = "\$t\$ [s]", ylabel = "\$y\$", label = ["\$y_{\\textrm{true}}\$" "\$\\hat{y}\$"], legend = :topright)
# display(f_y_2RLS)

e_θ_2RLS = [norm(θ - θ̂_dat_2RLS[k]) for k in 1:length(t)]
# f_e_θ_2RLS = plot(t, e_θ_2RLS, xlabel = "\$t\$ [s]", ylabel = "\$ e_{\\theta} = \\left| \\hat{\\theta} - \\theta\\right|\$", label = "2RLS", legend = :topright)
# display(f_e_θ_2RLS)

## summary
Fig_y = plot(t, [dat_RLS[1].ŷ_RLS dat_RLS[2].ŷ_RLS dat_RLS[3].ŷ_RLS dat_RLS[4].ŷ_RLS ŷ_2RLS y], xlabel="\$t\$ [s]", ylabel="\$y\$, \$\\hat{y}\$", label=["RLS: \\rho = 1" "RLS: \\rho = 10" "RLS: \\rho = 10^{2}" "RLS: \\rho = 10^{3}" "2RLS" "measured"], legend=:topright)
display(Fig_y)
savefig(Fig_y, "Fig_y.pdf")

Fig_e_y = plot(t, [dat_RLS[1].e_y_RLS dat_RLS[2].e_y_RLS dat_RLS[3].e_y_RLS dat_RLS[4].e_y_RLS e_y_2RLS], xlabel="\$t\$ [s]", ylabel="\$ e_{y} = \\vert \\hat{y} - y_{\\rm{true}}\\vert\$", label=["RLS: \\rho = 1" "RLS: \\rho = 10" "RLS: \\rho = 10^{2}" "RLS: \\rho = 10^{3}" "2RLS"], legend=:topright)
display(Fig_e_y)
savefig(Fig_e_y, "Fig_e_y.pdf")

Fig_e_θ = plot(t, [dat_RLS[1].e_θ_RLS dat_RLS[2].e_θ_RLS dat_RLS[3].e_θ_RLS dat_RLS[4].e_θ_RLS e_θ_2RLS], xlabel="\$t\$ [s]", ylabel="\$ e_{\\theta} = \\Vert \\hat{\\theta} - \\theta\\Vert\$", label=:false, legend=:topright)
display(Fig_e_θ)
savefig(Fig_e_θ, "Fig_e_theta.pdf")

Fig_e_y_e_θ = plot(Fig_e_y, Fig_e_θ, layout=(2, 1))
display(Fig_e_y_e_θ)
savefig(Fig_e_y_e_θ, "Fig_e_y_e_theta.pdf")

Fig_θ = plot(t, hcat(dat_RLS[1].θ̂_dat_RLS...)', layout=(4, 1), label=["RLS: \\rho = 1" :false :false :false], ylabel=["\$\\hat{\\theta}_{1}\$" "\$\\hat{\\theta}_{2}\$" "\$\\hat{\\theta}_{3}\$" "\$\\hat{\\theta}_{4}\$"], xlabel=["" "" "" "\$t\$ [s]"],  size=(600, 700))
plot!(Fig_θ, t, hcat(dat_RLS[2].θ̂_dat_RLS...)', label=["RLS: \\rho = 10" :false :false :false])
plot!(Fig_θ, t, hcat(dat_RLS[3].θ̂_dat_RLS...)', label=["RLS: \\rho = 10^{2}" :false :false :false])
plot!(Fig_θ, t, hcat(dat_RLS[4].θ̂_dat_RLS...)', label=["RLS: \\rho = 10^{3}" :false :false :false])
plot!(Fig_θ, t, hcat(θ̂_dat_2RLS...)', label=["2RLS" :false :false :false])
display(Fig_θ)
savefig(Fig_θ, "Fig_theta.pdf")