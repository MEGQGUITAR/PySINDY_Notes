using CairoMakie
using LaTeXStrings
using LinearAlgebra
using Printf
using Statistics
using DifferentialEquations
using ForwardDiff

# --- Funciones Auxiliares ---
function safe_sqrt(x)
    return sqrt(max(0, x))
end
Energy = [4, 4.009950493535854, 4.048808686896448, 4.0954429520014575,4.264814470207898, 4.413750407073517]#Energía minima de cada kappa k=0,0.01,0.05,0.1,0.3,0.5
#Limites de u0 para cada energía (1.2E,1.5E,2E,3E) y kappa
rmin_1 = [1.035,0.86,0.72,0.58]
rmax_1 = [1.94,2.3,2.74,3.42]
rmin_2 = [1.035,0.86,0.72,0.58]
rmax_2 = [1.93,2.3,2.72,3.38]
rmin_3 = [1,0.86,0.72,0.57]
rmax_3 = [1.9,2.24,2.64,3.24]
rmin_4 = [1,0.85,0.71,0.57]
rmax_4 = [1.9,2.2,2.56,3.1]
rmin_5 = [0.99,0.83,0.7,0.56]
rmax_5 = [1.8,2.05,2.36,2.8]
rmin_6 = [0.97,0.81,0.68,0.55]
rmax_6 = [1.7,1.95,2.24,2.62]
Energy_neg=[100.02000100020005, 20.10012562973092, 10.201020622952841,3.967216138563186]#k=-0.01,-0.05,-0.1,-0.3
#Limites de u0 para cada energía y kappa<0
rmin_neg_1 = [0.19,0.19,0.19]
rmax_neg_1 = [12.6,12.91,13.41]
rmin_neg_2 = [0.45,0.45,0.44]
rmax_neg_2 = [5.57,5.75,6]
rmin_neg_3 = [0.65,0.64,0.63]
rmax_neg_3 = [4,4.04,4.22]
rmin_neg_4 = [1.34,1.29,1.25]
rmax_neg_4 = [1.95,2.1,2.25]
# --- Parámetros del Sistema ---
En = 2*Energy[6] # Energía
k = 0.5   # Parámetro 'kappa'
t_span = (0.0, 5000.0)    #TIEMPO DE INTEGRACIÓN
p_points = 9
r_values=LinRange(rmin_6[3], rmax_6[3], 75)
# --- Definición de la estructura de Parámetros Mutables ---
if !@isdefined(SimParams)
    mutable struct SimParams
        k::Float64
        lyap_sum::Float64
        idx_trajectory::Int64
        t_final_integration::Float64
        u_0::Float64
        pu_0::Float64
        poincare_data_ref::Ref{Vector{Vector{Tuple{Float64, Float64, Float64}}}}
    end
end

# --- Ecuaciones Hamiltonianas ---
function hamilton_eqs!(du, u, p::SimParams, t)
    u1, v2, pu, pv = u

    du[1] = pu # dx/dt
    du[2] = pv# dy/dt

    du[3] = u1*((8*(u1^2+3*v2^2))/((u1^2-v2^2)^3) - p.k*(u1^2-v2^2) - 2)# dpx/dt
    du[4] = v2*(-(8*(v2^2+3*u1^2))/((u1^2-v2^2)^3) + p.k*(u1^2-v2^2) - 2)# dpy/dt
end

# --- Ecuaciones del Espacio Tangente (usando Diferenciación Automática) ---
function tangent_eqs_AD!(duv, uv, p::SimParams, t)
    R = view(uv, 1:4)
    U = view(uv, 5:8)

    f_hamilton = R_vec -> begin
        out_du = similar(R_vec)
        hamilton_eqs!(out_du, R_vec, p, t)
        return out_du
    end

    J = ForwardDiff.jacobian(f_hamilton, R)

    hamilton_eqs!(view(duv, 1:4), R, p, t)
    mul!(view(duv, 5:8), J, U)
end

# --- Callback de Renormalización de Gram-Schmidt ---
renorm_interval = 50.0

function lyapunov_callback(integrator)
    U = view(integrator.u, 5:8)
    norm_U = norm(U)

    if norm_U > 0
        U ./= norm_U
        integrator.p.lyap_sum += log(norm_U)
    end
    nothing
end

# --- Callback para la Sección de Poincaré ---
function condition_poincare(u, t, integrator)
    if t < 1e-2
        return 1.0   # cualquier valor ≠ 0 evita disparo
    end
    return u[2] # Condición: v = 0
end

function affect_poincare!(integrator)
    if integrator.u[4] > 0 #DIRECCIÓN POSITIVA pv > 0
        u1,v2,pu,pv= integrator.u[1:4]
        idx_trajectory = integrator.p.idx_trajectory
        current_lyap_sum = integrator.p.lyap_sum

        poincare_data_ref = integrator.p.poincare_data_ref[]
        while length(poincare_data_ref) < idx_trajectory
            push!(poincare_data_ref, [])
        end

        push!(poincare_data_ref[idx_trajectory], (u1, pu, current_lyap_sum))
    end
    nothing
end

function simulate_trajectory!(u_0, v_0, pu_0, pv_0, traj_idx,k_val, 
                              poincare_data_by_trajectory, t_span, callbacks_list)
    
    y0_state = [u_0, v_0, pu_0, pv_0]
    u0_tangent = [1.0, 0.0, 0.0, 0.0]
    u0_tangent_normalized = u0_tangent / norm(u0_tangent)
    uv0 = vcat(y0_state, u0_tangent_normalized)
    p_params = SimParams(k_val, 0.0, traj_idx, t_span[2], u_0, pu_0, Ref(poincare_data_by_trajectory))

    while length(poincare_data_by_trajectory) < traj_idx
        push!(poincare_data_by_trajectory, [])
    end
    prob = ODEProblem(tangent_eqs_AD!, uv0, t_span, p_params)
    sol = solve(prob, Tsit5(), reltol=1e-8, abstol=1e-8, callback=callbacks_list)
    
    return nothing
end
function run_simulation()
    println("Iniciando simulación...")
    poincare_data_by_trajectory = Vector{Vector{Tuple{Float64, Float64, Float64}}}()
    initial_conditions_list = []
    cb_poincare = ContinuousCallback(condition_poincare, affect_poincare!;
                                     affect_neg! = affect_poincare!,
                                     save_positions=(false, false),
                                     interp_points=50)
    
    cb_renorm = PeriodicCallback(lyapunov_callback, renorm_interval;
                                 initial_affect=true,
                                 save_positions=(false, false))
    
    callbacks_list = CallbackSet(cb_renorm, cb_poincare)
    global_traj_idx = 1
    for u_0 in r_values
        range=2*(En-u_0^2-4/u_0^2 -k/4*u_0^4)
        if range <0
            continue
        end
        p_values = LinRange(-safe_sqrt(range), safe_sqrt(range), p_points)
        for pu_0 in p_values
            v_0=0.0
            discriminant=2*(En-u_0^2-4/u_0^2 -k/4*u_0^4)-pu_0^2
            if discriminant < 0
                continue
            end
            pv_pos = safe_sqrt(discriminant)
            for pv_0 in [pv_pos]
                println("Simulando trayectoria $global_traj_idx con u_0 = $u_0, pv_0 = $pv_0")
                push!(initial_conditions_list, (global_traj_idx, u_0, pv_0))
                simulate_trajectory!(u_0, v_0, pu_0, pv_0, global_traj_idx, k, poincare_data_by_trajectory, t_span, callbacks_list)
                global_traj_idx += 1
            end
        end
    end

    println("\n--- Resumen de Resultados ---")
    total_trajectories = length(poincare_data_by_trajectory)
    total_poincare_points = sum(length.(poincare_data_by_trajectory))
    println("Número total de trayectorias simuladas: $total_trajectories")
    println("Total de cruces de Poincaré capturados: $total_poincare_points")
    all_q_lyap = Float64[]
    all_pp_lyap = Float64[]
    all_lambda_lyap = Float64[]
    trajectory_q0 = Float64[]  # Para almacenar y0 de cada trayectoria
    trajectory_pp0 = Float64[] # Para almacenar py0 de cada trayectoria
    trajectory_sign = Int64[]  # Para almacenar signo (+1 o -1)
    final_lambdas_by_trajectory = fill(NaN, total_trajectories)
    
    println("\n--- Análisis de Exponentes de Lyapunov ---")
    for idx in 1:total_trajectories
        points_data = poincare_data_by_trajectory[idx]
        
        if !isempty(points_data)
            final_lyap_sum_for_trajectory = points_data[end][3]
            lambda_val = final_lyap_sum_for_trajectory / t_span[2]
            final_lambdas_by_trajectory[idx] = lambda_val
            
            for (traj_idx, q0_init, pp0_init) in initial_conditions_list
                if traj_idx == idx
                    push!(trajectory_q0, q0_init)
                    push!(trajectory_pp0, pp0_init)
                    push!(trajectory_sign, pp0_init > 0 ? 1 : -1)
                    break
                end
            end
            
            if lambda_val < 5e-3 && lambda_val > 0
                println("Trayectoria $idx: λ = $(@sprintf("%.6f", lambda_val)) (cuasiperiódico/regular)")
            else
                println("Trayectoria $idx: λ = $(@sprintf("%.6f", lambda_val)) (caótico)")
            end
            
            for (q, pp, _) in points_data
                push!(all_q_lyap, q)
                push!(all_pp_lyap, pp)
                push!(all_lambda_lyap, lambda_val)
            end
        end
    end
    
    valid_lambdas = filter(!isnan, final_lambdas_by_trajectory)
    if !isempty(valid_lambdas)
        min_lambda = minimum(valid_lambdas)
        max_lambda = maximum(valid_lambdas)
        println("Rango de exponentes de Lyapunov calculados: [$(@sprintf("%.4f", min_lambda)), $(@sprintf("%.4f", max_lambda))]")
    else
        println("No se calcularon exponentes de Lyapunov válidos.")
    end
    
    # --- Generación de Gráficos con CairoMakie ---
    
    if total_poincare_points == 0
        println("\nNo hay puntos válidos en la sección de Poincaré para graficar.")
        return poincare_data_by_trajectory, all_q_lyap, all_pp_lyap, all_lambda_lyap
    end
    # --- Gráfica 1: Sección de Poincaré con CairoMakie ---
    total_trajectories = length(poincare_data_by_trajectory)
    all_q = Float64[]
    all_parallel_p = Float64[]
    all_traj_idx = Int64[]
    for idx in 1:total_trajectories
        points_data = poincare_data_by_trajectory[idx]
        for (q, pp, _) in points_data
            push!(all_q, q)
            push!(all_parallel_p, pp)
            push!(all_traj_idx, idx)
        end
    end
    if !isempty(all_q)
        fig1 = Figure(size=(1200, 900))
        ax1 = Axis(fig1[1, 1],
            xlabel = L"u",
            ylabel = L"P_u",
            xlabelsize = 75,
            ylabelsize = 75,
            xticklabelsize = 35,
            yticklabelsize = 35,
            titlesize = 2,
            title = "",
            #autolimitaspect=1,# Mantener aspecto 1:1
            xgridvisible=false,
            ygridvisible=false
        )
        traj_colors =  cgrad(:managua, total_trajectories, categorical=true)

        scatter!(ax1, all_q, all_parallel_p, color = all_traj_idx,
        colormap = traj_colors,
        colorrange = (1, total_trajectories),  # mapea índices enteros a colores
        markersize = 1.5,
        strokewidth = 0,
        alpha = 0.7)
        save("Poincare_section_E=$(En)_kappa=$(k).png", fig1, px_per_unit=3,dpi=300)
        println("Gráfico de la sección de Poincaré guardado como PNG con CairoMakie.")
    else
        println("No hay puntos válidos en la sección de Poincaré para graficar.")
    end
    # --- Gráfica 2: Exponente de Lyapunov con CairoMakie ---
    if !isempty(all_q_lyap)
        filtered_x = all_q_lyap
        filtered_py = all_pp_lyap
        filtered_lambda = all_lambda_lyap
        
        color_lims = (0,0.25) # AJUSTA EL LIMITE DE LA BARRA DE COLOR PARA EL LLE
        
        # Crear figura con CairoMakie
        fig = Figure(size=(1400,1000))

        ax = Axis(fig[1,1],width=900,
            xlabel=L"u",
            ylabel=L"P_u",
            xlabelsize=75,
            ylabelsize=75,
            xticklabelsize=35,
            yticklabelsize=35,
            xgridvisible=false,
            ygridvisible=false
        )

        scatter = scatter!(
            ax,
            filtered_x,
            filtered_py,
            color=filtered_lambda,
            colormap=:hot,
            colorrange=color_lims,
            markersize=1.2
        )
        #Colorbar(fig[1,2], scatter,label=L"\lambda",labelsize=65,ticklabelsize=45)  #BARRA DE COLOR
        Box(fig[1,2], width=200, height=800, color=:transparent, strokewidth=0)

        
        save("Poincare_section_E=$(En)_kappa=$(k)_Lyapunov.png", fig, px_per_unit=3,dpi=300)
        println("Gráfico con exponentes de Lyapunov guardado como PNG con CairoMakie.")
        
    else
        println("No hay suficientes datos para generar la gráfica del Exponente de Lyapunov.")
    end
    
    println("\n--- Simulación completada ---")
    return poincare_data_by_trajectory, all_q_lyap, all_pp_lyap, all_lambda_lyap
end

# Ejecutar simulación
poincare_data_by_trajectory, all_q_lyap, all_pp_lyap, all_lambda_lyap = run_simulation()