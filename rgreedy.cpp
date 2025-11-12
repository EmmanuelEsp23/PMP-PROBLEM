#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <cmath>
#include <random>
#include <chrono>
#include <numeric>
#include <limits>
#include <iomanip>

// Este código lo realicé utilizando Gemini con C++14, adaptado de Python para mayor eficiencia, pero codifiqué 
// en Python porque es un lenguaje que manejo mejor. Espero que esté bien así, cualquier duda estoy a sus órdenes.
using DistanceMatrix = std::vector<std::vector<double>>;

// --- INICIO DE LA CORRECCIÓN DE ALEATORIEDAD ---
// std::random_device no es confiable en MinGW, siempre da la misma semilla.
// Usamos el reloj de alta resolución (nanosegundos) para obtener una semilla única.
unsigned long long seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
std::mt19937 rng(seed);
// --- FIN DE LA CORRECCIÓN DE ALEATORIEDAD ---


// --- 1. LECTURA Y CÁLCULO BÁSICO ---

bool read_instance_from_file(const std::string& filename, int& n, int& p, DistanceMatrix& dist_matrix) {
    std::cout << "--- Reading instance " << filename << " ---" << std::endl;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error reading file " << filename << std::endl;
        return false;
    }
    std::string line;
    if (!std::getline(file, line)) return false; n = std::stoi(line);
    if (!std::getline(file, line)) return false; p = std::stoi(line);

    dist_matrix.resize(n, std::vector<double>(n));
    for (int i = 0; i < n; ++i) {
        if (!std::getline(file, line)) return false;
        std::stringstream ss(line);
        for (int j = 0; j < n; ++j) {
            if (!(ss >> dist_matrix[i][j])) return false;
        }
    }
    return true;
}

double calculate_total_cost(const std::vector<int>& solution, int n, const DistanceMatrix& dist_matrix) {
    double total_cost = 0.0;
    if (solution.empty()) return std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
        double min_dist = std::numeric_limits<double>::infinity();
        for (int med : solution) {
            min_dist = std::min(min_dist, dist_matrix[i][med]);
        }
        total_cost += min_dist;
    }
    return total_cost;
}

// --- 2. HEURÍSTICA CONSTRUCTIVA (RGREEDY RÁPIDO) ---

std::pair<std::vector<int>, double> rgreedy_constructor(int n, int p, const DistanceMatrix& dist_matrix, int k_rcl) {
    std::cout << "\n--- Running rgreedy (FAST implementation, k_rcl = " << k_rcl << ") ---" << std::endl;
    std::vector<int> solution;
    solution.reserve(p); // Reservar espacio
    std::vector<int> candidates(n);
    std::iota(candidates.begin(), candidates.end(), 0);

    std::vector<double> min_dists(n, std::numeric_limits<double>::infinity());
    double final_cost = std::numeric_limits<double>::infinity();

    for (int i = 0; i < p; ++i) {
        std::vector<std::pair<double, int>> candidate_costs;
        candidate_costs.reserve(candidates.size());

        for (int c : candidates) {
            double new_total_cost = 0.0;
            for (int j = 0; j < n; ++j) {
                new_total_cost += std::min(min_dists[j], dist_matrix[j][c]);
            }
            candidate_costs.push_back({new_total_cost, c});
        }

        std::sort(candidate_costs.begin(), candidate_costs.end());

        int rcl_size = std::min((int)candidate_costs.size(), k_rcl);
        if (rcl_size == 0) break; // No quedan candidatos

        std::uniform_int_distribution<int> dist(0, rcl_size - 1);
        std::pair<double, int> choice = candidate_costs[dist(rng)];
        
        final_cost = choice.first;
        int chosen_node = choice.second;

        solution.push_back(chosen_node);
        candidates.erase(std::remove(candidates.begin(), candidates.end(), chosen_node), candidates.end());

        for (int j = 0; j < n; ++j) {
            min_dists[j] = std::min(min_dists[j], dist_matrix[j][chosen_node]);
        }
        
        std::cout << "    -> Node chosen: " << chosen_node << " (New Total Cost: " << final_cost << ")" << std::endl;
    }
    return {solution, final_cost};
}

// --- 3. BÚSQUEDA LOCAL (RÁPIDA Y CORREGIDA) ---

void update_structures(const std::vector<int>& solution, int n, const DistanceMatrix& dist_matrix,
                       std::vector<int>& closest, std::vector<int>& sec_closest, double& current_cost) {
    current_cost = 0.0;
    for (int i = 0; i < n; ++i) {
        double d1 = std::numeric_limits<double>::infinity();
        double d2 = std::numeric_limits<double>::infinity();
        int m1 = -1, m2 = -1;

        for (int med : solution) {
            double dist = dist_matrix[i][med];
            if (dist < d1) {
                d2 = d1; m2 = m1;
                d1 = dist; m1 = med;
            } else if (dist < d2) {
                d2 = dist; m2 = med;
            }
        }
        closest[i] = m1;
        sec_closest[i] = m2; // m2 será -1 si p=1, lo cual es correcto
        current_cost += d1;
    }
}

std::pair<std::vector<int>, double> local_search_fast(const std::vector<int>& initial_solution, int n, const DistanceMatrix& dist_matrix) {
    std::vector<int> current_solution = initial_solution;
    double current_cost;
    std::vector<int> closest(n), sec_closest(n);
    
    std::vector<bool> in_solution(n, false);
    for (int med : current_solution) {
        in_solution[med] = true;
    }

    update_structures(current_solution, n, dist_matrix, closest, sec_closest, current_cost);
    std::cout << "\n--- Starting Fast Local Search (Initial Cost: " << current_cost << ") ---" << std::endl;

    while (true) {
        bool improvement_found = false;
        
        // BUCLE CORREGIDO: Usar un índice para evitar la invalidación del iterador
        for (size_t i = 0; i < current_solution.size(); ++i) {
            int m_out = current_solution[i];
            
            for (int m_in = 0; m_in < n; ++m_in) {
                if (in_solution[m_in]) continue; // Solo probar nodos que NO están en la solución

                double delta = 0.0;
                for (int j = 0; j < n; ++j) { // Bucle sobre todos los clientes
                    double dist_current = dist_matrix[j][closest[j]];
                    double dist_in = dist_matrix[j][m_in];
                    
                    if (closest[j] == m_out) {
                        // CORRECCIÓN DE BUG: Comprobar si sec_closest es válido
                        double dist_sec = (sec_closest[j] != -1) 
                                          ? dist_matrix[j][sec_closest[j]] 
                                          : std::numeric_limits<double>::infinity();
                        
                        double new_dist = std::min(dist_sec, dist_in);
                        delta += (new_dist - dist_current);
                    } else {
                        double new_dist = std::min(dist_current, dist_in);
                        delta += (new_dist - dist_current);
                    }
                }

                // Estrategia First Improvement
                if (delta < -1e-9) { 
                    // --- Aplicar el swap ---
                    current_solution[i] = m_in; // Reemplazar m_out con m_in (más seguro)
                    in_solution[m_out] = false;
                    in_solution[m_in] = true;
                    
                    current_cost += delta;
                    improvement_found = true;
                    
                    std::cout << "    -> Quick swap: " << m_out << " (out) <-> " << m_in << " (in) | New cost: " << current_cost << std::endl;
                    
                    // --- Reconstruir estructuras ---
                    double current_cost_check; // Variable temporal
                    update_structures(current_solution, n, dist_matrix, closest, sec_closest, current_cost_check);
                    
                    if (std::abs(current_cost - current_cost_check) > 1e-5) {
                         std::cout << "Warning: Delta drift detected. Resetting cost to " << current_cost_check << std::endl;
                         current_cost = current_cost_check;
                    }
                         
                    break; // Salir del bucle m_in
                }
            } // fin del bucle m_in
            
            if (improvement_found) {
                break; // Salir del bucle m_out (para reiniciar el bucle while)
            }
        } // fin del bucle m_out

        if (!improvement_found) {
            std::cout << "  > Local optimum reached." << std::endl;
            break; // Salir del bucle while(true)
        }
    }
    return {current_solution, current_cost};
}

// --- 4. FUNCIÓN MAIN ---

int main() {
    std::cout << std::fixed << std::setprecision(6);

    std::string filename;
    std::cout << "Enter instance file path: ";
    std::cin >> filename;

    int n, p;
    DistanceMatrix dist_matrix;
    if (!read_instance_from_file(filename, n, p, dist_matrix)) {
        return 1;
    }

    std::cout << "\n--- Data: n=" << n << ", p=" << p << " ---" << std::endl;

    auto total_start_time = std::chrono::high_resolution_clock::now();

    // 1. RGREEDY
    auto constructive_start_time = std::chrono::high_resolution_clock::now();
    
    // =================================================================
    // CORRECCIÓN 1: Reemplazar 'auto [var1, var2]' con 'std::pair'
    // =================================================================
    std::pair<std::vector<int>, double> result_constructive = rgreedy_constructor(n, p, dist_matrix, 3);
    std::vector<int> sol_constructive = result_constructive.first;
    double cost_constructive = result_constructive.second;
    // =================================================================

    auto constructive_end_time = std::chrono::high_resolution_clock::now();
    
    std::cout << "Constructive Cost: " << cost_constructive << std::endl;

    // 2. FAST LOCAL SEARCH
    auto ls_start_time = std::chrono::high_resolution_clock::now();

    // =================================================================
    // CORRECCIÓN 2: Reemplazar 'auto [var1, var2]' con 'std::pair'
    // =================================================================
    std::pair<std::vector<int>, double> result_final = local_search_fast(sol_constructive, n, dist_matrix);
    std::vector<int> sol_final = result_final.first;
    double cost_final = result_final.second;
    // =================================================================
    
    auto ls_end_time = std::chrono::high_resolution_clock::now();

    auto total_end_time = std::chrono::high_resolution_clock::now();

    // --- IMPRESIÓN DE RESULTADOS ---
    std::cout << "\n=== FINAL RESULT ===" << std::endl;
    std::cout << "Initial Cost: " << cost_constructive << " -> Final Cost: " << cost_final << std::endl;
    std::cout << "Improvement: " << (cost_constructive - cost_final) << std::endl;
    
    std::cout << "Final Medians: [";
    std::sort(sol_final.begin(), sol_final.end());
    for (size_t i = 0; i < sol_final.size(); ++i) {
        std::cout << sol_final[i] << (i == sol_final.size() - 1 ? "" : ", ");
    }
    std::cout << "]" << std::endl;

    // --- IMPRESIÓN DE TIEMPOS ---
    std::chrono::duration<double> constructive_secs = constructive_end_time - constructive_start_time;
    std::chrono::duration<double> ls_secs = ls_end_time - ls_start_time;
    std::chrono::duration<double> total_secs = total_end_time - total_start_time;

    std::cout << "\n--- EXECUTION TIME ---" << std::endl;
    std::cout << "Constructive (rgreedy): " << constructive_secs.count() << " seconds" << std::endl;
    std::cout << "Local Search (Fast):    " << ls_secs.count() << " seconds" << std::endl;
    std::cout << "Total Time:             " << total_secs.count() << " seconds" << std::endl;

    // --- PAUSA PARA VER RESULTADOS ---
    std::cout << "\nPress Enter to exit...";
    // Limpia cualquier 'Enter' fantasma que haya quedado en el buffer de entrada
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n'); 
    // Espera a que el usuario presione Enter
    std::cin.get(); 
    // --- FIN PAUSA ---

    return 0;
}