using Plots
gr()
using LaTeXStrings
using PlotThemes
using LinearAlgebra
using Printf
using Statistics
using DifferentialEquations
using ForwardDiff

# --- Funciones Auxiliares ---
function safe_sqrt(x)
    return sqrt(max(0, x))
end

# --- Parámetros del Sistema ---
En = 4 # Energía minima=2/a
a = 1   # Parámetro 'a'
b = 1.5 # Parámetro 'b'
# r_values son los valores iniciales de y_0 para las diferentes trayectorias
r_values = LinRange(-3.52, 3.52, 90)
p_points=7 # Número de puntos de px_0 para cada y_0
# --- Definición de la estructura de Parámetros Mutables ---
if !@isdefined(SimParams)
    mutable struct SimParams
        a::Float64
        b::Float64
        lyap_sum::Float64
        idx_trajectory::Int64
        t_final_integration::Float64
        y0::Float64
        py0::Float64
        poincare_data_ref::Ref{Vector{Vector{Tuple{Float64, Float64, Float64}}}}
    end
end

# --- Ecuaciones Hamiltonianas ---
function hamilton_eqs!(du, u, p::SimParams, t)
    x, y, px, py = u

    denom_arg1 = p.a^2 + 2p.a*x + x^2 + y^2
    denom_arg2 = p.a^2 - 2p.a*x + x^2 + y^2

    den1 = (safe_sqrt(denom_arg1))^3
    den2 = (safe_sqrt(denom_arg2))^3

    den1_inv = den1 > 1e-10 ? 1.0 / den1 : 0.0
    den2_inv = den2 > 1e-10 ? 1.0 / den2 : 0.0

    du[1] = px + p.b*y/2
    du[2] = py - p.b*x/2

    du[3] = (p.b*(2py - p.b*x)/2 + (2x + 2p.a)*den1_inv + (2x - 2p.a)*den2_inv) / 2
    du[4] = (p.b*(-2px - p.b*y)/2 + (2y)*den2_inv + (2y)*den1_inv) / 2
end

# --- Calculo del Jacobiano del sistema (usando Diferenciación Automática) ---
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

# --- Renormalización de Gram-Schmidt y calculo de exponentes de Lyapunov con el algoritmo de Benetti ---
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
    return u[1]  # Condition: x = 0
end

function affect_poincare!(integrator)
    # Orientación del mapa y evitar doble conteo (x=0 & px > 0)
    if abs(integrator.u[1]) <1e-2 && integrator.u[3] > b*integrator.u[2]/2
        current_y = integrator.u[2] # Valor de Y válido
        current_py = integrator.u[4]# Valor de PY válido
        idx_trajectory = integrator.p.idx_trajectory
        current_lyap_sum = integrator.p.lyap_sum
        poincare_data_ref = integrator.p.poincare_data_ref[]
        while length(poincare_data_ref) < idx_trajectory
            push!(poincare_data_ref, [])
        end
        push!(poincare_data_ref[idx_trajectory], (current_y, current_py, current_lyap_sum))
    end
    nothing
end
function simulate_trajectory!(x_0, y_0, px_0, py_0, traj_idx, a_val, b_val, 
                              poincare_data_by_trajectory, t_span, callbacks_list)
    y0_state = [x_0, y_0, px_0, py_0]
    u0_tangent = [1.0, 0.0, 0.0, 0.0]
    u0_tangent_normalized = u0_tangent / norm(u0_tangent)
    uv0 = vcat(y0_state, u0_tangent_normalized)
    p_params = SimParams(a_val, b_val, 0.0, traj_idx, t_span[2], y_0, py_0, Ref(poincare_data_by_trajectory))
    while length(poincare_data_by_trajectory) < traj_idx
        push!(poincare_data_by_trajectory, [])
    end
    prob = ODEProblem(tangent_eqs_AD!, uv0, t_span, p_params)
    sol = solve(prob, Tsit5(), reltol=1e-6, abstol=1e-8, callback=callbacks_list)
    return nothing
end

function run_simulation()
    println("Iniciando simulación...")
    println("Cada y_0 generará 2 trayectorias (p_y positivo y negativo) cuando sea posible")
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
    
    t_span = (0.0, 1000.0)    #TIEMPO DE INTEGRACIÓN
    global_traj_idx = 1
    for y_0 in r_values
        x_0 = 0.0
        discriminant = 2*En - (b^2*y_0^2)/4 - 4/safe_sqrt(a^2 + y_0^2) #discriminante para px_0 
        if discriminant < 0
            println("Condición física no satisfecha para y_0 = $y_0. Saltando esta condición inicial.")
            continue
        end
        PX_range=LinRange(-safe_sqrt(discriminant), safe_sqrt(discriminant), p_points)
        for px_0 in PX_range
            discriminant_trajectory = 2*En - (b^2*y_0^2)/4 - 4/safe_sqrt(a^2 + y_0^2) - px_0^2 # discriminante para py_0 para cualquier valor de px_0 permitido
            if discriminant_trajectory < 0
                println("Condición física no satisfecha para y_0 = $y_0, px_0 = $px_0. Saltando esta condición inicial.")
                continue
            end
            #Py_0 positivo y negativo (siempre que el discriminante sea positivo)
            py_0_pos = safe_sqrt(discriminant_trajectory)
            if py_0_pos > 0 
                push!(initial_conditions_list, (global_traj_idx, y_0, py_0_pos))
                simulate_trajectory!(x_0, y_0, px_0, py_0_pos, global_traj_idx, a, b,
                                    poincare_data_by_trajectory, t_span, callbacks_list)
                global_traj_idx += 1
            end
            if discriminant_trajectory > 0  # Solo si no es cero
                py_0_neg = -safe_sqrt(discriminant_trajectory)
                push!(initial_conditions_list, (global_traj_idx, y_0, py_0_neg))
                simulate_trajectory!(x_0, y_0, px_0, py_0_neg, global_traj_idx, a, b,
                                    poincare_data_by_trajectory, t_span, callbacks_list)
                global_traj_idx += 1
            end
        end
    end
    
    println("\n--- Resumen de Resultados ---")
    total_trajectories = length(poincare_data_by_trajectory)
    total_poincare_points = sum(length.(poincare_data_by_trajectory))
    println("Número total de trayectorias simuladas: $total_trajectories")
    println("Total de cruces de Poincaré capturados: $total_poincare_points")
    all_y_lyap = Float64[]
    all_py_lyap = Float64[]
    all_lambda_lyap = Float64[]
    trajectory_y0 = Float64[]  # Para almacenar y0 de cada trayectoria
    trajectory_py0 = Float64[] # Para almacenar py0 de cada trayectoria
    trajectory_sign = Int64[]  # Para almacenar signo (+1 o -1)
    final_lambdas_by_trajectory = fill(NaN, total_trajectories)
    
    println("\n--- Análisis de Exponentes de Lyapunov ---")
    for idx in 1:total_trajectories
        points_data = poincare_data_by_trajectory[idx]
        
        if !isempty(points_data)
            final_lyap_sum_for_trajectory = points_data[end][3]
            lambda_val = final_lyap_sum_for_trajectory / t_span[2]
            final_lambdas_by_trajectory[idx] = lambda_val
            
            for (traj_idx, y0_init, py0_init) in initial_conditions_list
                if traj_idx == idx
                    push!(trajectory_y0, y0_init)
                    push!(trajectory_py0, py0_init)
                    push!(trajectory_sign, py0_init > 0 ? 1 : -1)
                    break
                end
            end
            
            if lambda_val < 5e-3 && lambda_val > 0
                println("Trayectoria $idx: λ = $(@sprintf("%.6f", lambda_val)) (cuasiperiódico/regular)")
            end
            
            for (y, py, _) in points_data
                push!(all_y_lyap, y)
                push!(all_py_lyap, py)
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
    
    # --- Generación de Gráficos ---
    
    if total_poincare_points == 0
        println("\nNo hay puntos válidos en la sección de Poincaré para graficar.")
        return poincare_data_by_trajectory, all_y_lyap, all_py_lyap, all_lambda_lyap
    end
    
    # --- Gráfica 1: Sección de Poincaré ---
    total_trajectories_for_plot = length(poincare_data_by_trajectory)
    colors_map_2d_ci = Plots.cgrad(:rainbow, total_trajectories_for_plot, categorical=true)
    
    p_poincare_by_ci_2d = Plots.plot(
       xlabel=L"y", 
       ylabel=L"p_y",
       title="",
       size=(1300, 900),
       dpi=600,
       left_margin=25Plots.mm,
       right_margin=10Plots.mm,
       top_margin=5Plots.mm,
       bottom_margin=5Plots.mm,
       legend=false,
       grid=false,
       # Fuentes grandes para revista
       xlabelfontsize=40,      # Aumentado de 16 a 24
       ylabelfontsize=40,      # Aumentado de 16 a 24
       xtickfontsize=30,       # Aumentado de 14 a 20
       ytickfontsize=30,       # Aumentado de 14 a 20
       titlefontsize=2,       # Aumentado de 16 a 28
       framestyle=:box,
       tick_direction=:out
    )
    
    for idx in 1:total_trajectories_for_plot
        points_data = poincare_data_by_trajectory[idx]
        if !isempty(points_data)
            y_coords = [pt[1] for pt in points_data]
            py_coords = [pt[2] for pt in points_data]
            
            current_color = colors_map_2d_ci[idx]
            
            Plots.scatter!(p_poincare_by_ci_2d, y_coords, py_coords,
                           color=current_color,
                           seriesalpha=0.7,
                           marker=:circle,
                           markersize=1,
                           markerstrokewidth=0,
                           label="")
        end
    end
    
    Plots.savefig(p_poincare_by_ci_2d, "Poincare_section_E=$(En)_B=$(b)_a=$(a).png")
    println("\nGráfico de la sección de Poincaré guardado como PNG.")
    
    # --- Gráfica 2: Exponente de Lyapunov (CORREGIDA) ---
    # Alternativa: Crear el gráfico de manera diferente
    if !isempty(all_y_lyap)
    # Filtrar valores
        valid_indices = findall(x -> -10 < x < 10, all_py_lyap)
        if length(valid_indices) > 0
            filtered_y = all_y_lyap[valid_indices]
            filtered_py = all_py_lyap[valid_indices]
            filtered_lambda = all_lambda_lyap[valid_indices]
        else
            filtered_y = all_y_lyap
            filtered_py = all_py_lyap
            filtered_lambda = all_lambda_lyap
        end
        color_lims = (0.0, 0.2)
        # Crear ticks personalizados
        custom_ticks = [0.0, 0.1, 0.2]
        # Crear el gráfico de forma más directa
        p_poincare_lyapunov_2d = Plots.scatter(
            filtered_y, filtered_py,
            marker=:circle,
            markersize=1,
            markerstrokewidth=0,
            marker_z=filtered_lambda,
            seriesalpha=0.7,
            label="",
            color=:haline,
            clims=color_lims,
            colorbar=true,
            colorbar_title="",
            colorbar_titlefontsize=35,
            colorbar_tickfontsize=25,
            colorbar_width=25,
            colorbar_tick=custom_ticks,  # Esto debería funcionar aquí
            # Configuración del gráfico
            xlabel=L"y", 
            ylabel=L"p_y",
            title="",
            size=(1300, 900),
            dpi=300,
            top_margin=5Plots.mm,
            left_margin=10Plots.mm,
            right_margin=35Plots.mm,
            bottom_margin=5Plots.mm,
            legend=false,
            grid=false,
            xlabelfontsize=40,
            ylabelfontsize=40,
            xtickfontsize=30,
            ytickfontsize=30,
            framestyle=:box,
            tick_direction=:out
        )
        # Anotación
        #annotate!(p_poincare_lyapunov_2d, maximum(filtered_y)*1.28, 0.0,text(L"\lambda", 35, :black, :center, rotation=90))
    
        Plots.savefig(p_poincare_lyapunov_2d, "Poincare_section_E=$(En)_B=$(b)_a=$(a)_Lyapunov.png")
        println("Gráfico con exponentes de Lyapunov guardado como PNG.")
    else
        println("No hay suficientes datos para generar la gráfica del Exponente de Lyapunov.")
    end
    println("\n--- Simulación completada ---")
    return poincare_data_by_trajectory, all_y_lyap, all_py_lyap, all_lambda_lyap
end

poincare_data_by_trajectory, all_y_lyap, all_py_lyap, all_lambda_lyap = run_simulation()