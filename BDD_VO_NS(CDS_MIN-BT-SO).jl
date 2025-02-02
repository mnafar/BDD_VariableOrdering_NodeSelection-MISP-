### A Pluto.jl notebook ###
# v0.16.0

using Markdown
using InteractiveUtils

# ╔═╡ 504a3e40-e0b5-11ef-0ef4-fb7ffa0af109


module DD

using DataStructures
using AutoHashEquals
using DelimitedFiles
using DataFrames
using CSV


# A structure to implementation an "arc" in DD; an arc has three parts (i.e. tail is the state from which the arc comes out, decision which is to store the decision on variables, value which is associated with the cost function in DD). 
@auto_hash_equals struct Arc{S,D,V}
    tail::S
	decision::D
    value::V
	
end

function Base.isapprox(x::Arc{S,D,V}, y::Arc{S,D,V}) where {S,D,V}
    return x.tail == y.tail && x.decision == y.decision && x.value ≈ y.value 
end


@auto_hash_equals struct Node{S,D,V}
    obj::V
	inarc::Union{Arc{S,D,V},Nothing}
    exact::Bool
	
	function Node{S,D,V}(obj, inarc=nothing, exact=true) where {S,D,V} 
        new(obj, inarc, exact) 
    end
end


function Base.isapprox(x::Node{S,D,V}, y::Node{S,D,V}) where {S,D,V}
    return x.obj ≈ y.obj && x.inarc ≈ y.inarc && x.exact == y.exact 
end


@auto_hash_equals struct Layer{S,D,V}
    nodes::Dict{S,Node{S,D,V}}
    exact::Bool
	
    function Layer{S,D,V}(nodes=Dict(), exact=true) where {S,D,V}
        return new(nodes, exact)
    end
end

function Base.iterate(layer::Layer{S,D,V}) where {S,D,V}
    return iterate(layer.nodes)
end

function Base.iterate(layer::Layer{S,D,V}, state) where {S,D,V}
    return iterate(layer.nodes, state)
end

function Base.length(layer::Layer{S,D,V}) where {S,D,V}
    return length(layer.nodes)
end

function Base.haskey(layer::Layer{S,D,V}, state::S) where {S,D,V}
    return haskey(layer.nodes, state)
end

function Base.getindex(layer::Layer{S,D,V}, state::S) where {S,D,V}
    return getindex(layer.nodes, state)
end

function Base.setindex!(
    layer::Layer{S,D,V}, node::Node{S,D,V}, state::S
) where {S,D,V}
    return setindex!(layer.nodes, node, state)
end


@auto_hash_equals struct Diagram{S,D,V}
    partial_sol::Vector{Int64}
    layers::Vector{Layer{S,D,V}}
    variables::Vector{Int64}
end

function Diagram(initial::Layer{S,D,V}) where {S,D,V}
    return Diagram{S,D,V}([], [initial], [])
end

function Diagram(instance)
    state = initial_state(instance)
    S = typeof(state)
    D = domain_type(instance)
    V = value_type(instance)
    node = Node{S,D,V}(zero(V))#, nothing, nothing, [], true)
    root = Layer{S,D,V}(Dict(state => node))
    return Diagram(root)
end

@auto_hash_equals struct Solution{D,V}
    decisions::Vector{D}  # for all variables, order 1:n
    objective::V
end

@auto_hash_equals struct Subproblem{S,D,V}
    # partial solution (assigned so far, in given order)
    variables::Vector{Int64}
    decisions::Vector{D}
    obj::V

    # state (to complete solution)
    state::S
end

######  interfaces

function initial_state() end
function domain_type end
function value_type end
function transitions end


######   generic functions

# This function gets the current decision variable and builds its corresponding layer of the decision diagram.

function build_layer(instance, diagram::Diagram{S,D,V}, variable) where {S,D,V}
    layer = Layer{S,D,V}()

    # Collect new states
	for (state, node) in diagram.layers[end] 
		for (arc, new_state) in transitions(instance, state, variable)
			if !haskey(layer, new_state) 
				layer[new_state] = Node{S,D,V}(node.obj + arc.value, arc, true)
			else
				if layer[new_state].obj < node.obj + arc.value 
					layer[new_state] = Node{S,D,V}(node.obj + arc.value, arc, true)
				end	
			end
		end
    end
    return layer
end



### this function builds a relaxed decision diagram which provides a dual bound

function top_down_DD_dual!(diagram::Diagram{S,D,V}, instance; W::Int64, variable_order::AbstractString, node_selection::AbstractString) where {S,D,V}
    
	@assert length(diagram.layers) == 1  	
	
	@assert (variable_order == "CDS" || variable_order == "MIN")
	
	@assert (node_selection == "SO" || node_selection == "BT")
		
	variables = sort!([i for i in 1:length(instance)], by=tup->sum(instance.graph[tup]), rev=false)

	counter = 1
	while length(diagram.variables) <= length(instance)
		variable = next_variable(instance.graph, diagram.layers[end], variable_order, [z for z in variables])
		if variable == nothing
			break
		end
		filter!(e->e≠variable, variables)		
		layer = build_layer(instance, diagram, variable)
		if length(layer) > W
			layer = approximate_DD(layer, W, node_selection)
		end
		push!(diagram.layers, layer)
		push!(diagram.variables, variable)
		counter += 1
		
	end
		
	@assert length(instance) + 1 == length(diagram.layers)
	### create terminal node	
	diagram.layers[end] = last_layer_into_terminal(diagram.layers[end]) 
	terminal = only(values(diagram.layers[end].nodes))
	return terminal.obj
	
end


### this function builds a restricted decision diagram which provides a primal bound
function top_down_DD_primal!(diagram::Diagram{S,D,V}, instance; W::Int64) where {S,D,V}
    
	@assert length(diagram.layers) == 1  	
		
	variables = sort!([i for i in 1:length(instance)], by=tup->sum(instance.graph[tup]), rev=false)
	
	for	variable in variables 
		layer = build_layer(instance, diagram, variable)
		if length(layer) > W
			collection = collect(layer)
			sort!(collection, by=tup->tup.second.obj, rev=true)		
			layer = Layer{S,D,V}(Dict(collection[1:W]), false)			
		end
		push!(diagram.layers, layer)
		push!(diagram.variables, variable)
	end
	
	@assert length(instance.graph) + 1 == length(diagram.layers)
	### create terminal node	
	diagram.layers[end] = last_layer_into_terminal(diagram.layers[end]) 
	terminal = only(values(diagram.layers[end].nodes))
	return terminal.obj
	
end

#################################################


### this function assigns the next variable when top-down compiling the decision diagram in a dynamic manner according to the given variable ordering heuristic, i.e. MIN, CDS

function next_variable(graph, layer::Layer{S,D,V}, ordering, unfixed) where {S,D,V}
	
	@assert (ordering == "MIN" || ordering == "CDS")
	
	states = keys(layer.nodes)
	variables = Dict([x=>0 for x in unfixed])	
		
	if ordering == "CDS"	
		for state in states
			if !(isempty(state))
				members = collect(state)
				for ind_v1 in 1:length(state)-1
					for ind_v2 in ind_v1+1:length(state)
						if graph[members[ind_v1]][members[ind_v2]] == 1
							variables[members[ind_v1]] += 1
							variables[members[ind_v2]] += 1
						end
					end
				end
			end
		end

	elseif ordering == "MIN"	

		for vertex in keys(variables)
			for state in states
				if vertex in state
					variables[vertex] += 1
				end
			end
		end		
	end
			
	variables_ordered = collect(keys(variables))
	
	sort!(variables_ordered, by=tup->variables[tup], rev=false)
		
	if length(variables_ordered) != 0
		return variables_ordered[1] 
	else 
		return nothing
	end	
	
end


### this function relaxes a given layer of the decision diagram (i.e. a layer whose width ecceeds the given maximum width) according to the given node selection heuristic, i.e. SO, BT

function approximate_DD(layer::Layer{S,D,V}, W, selection) where {S,D,V}	
	
	@assert (selection == "SO" || selection == "BT")
	
	collection = collect(layer)
	sort!(collection, by=tup->tup.second.obj, rev=true)		

	if selection == "SO"
		new_layer = Layer{S,D,V}(Dict(collection[1:W-1]), false)			
		merged_state, merged_node = merge([x.first for x in collection[W:end]], layer)
		if !haskey(new_layer, merged_state)
			new_layer[merged_state] = merged_node
		else				
			if new_layer[merged_state].obj < merged_node.obj
				new_layer[merged_state] = merged_node
			end				
		end
		
	elseif selection == "BT"
		
		if collection[W].second.obj == collection[W-1].second.obj
			value = collection[W-1].second.obj
			front = 1
			back = 1
			Tie_set = Set([collection[W-1].first, collection[W].first])
			for i in 2:W-1
				if value == collection[W-i].second.obj
					push!(Tie_set, collection[W-i].first)
					back += 1
				end
			end
		
			for i in 1:length(collection)-W
				if value == collection[W+i].second.obj
					push!(Tie_set, collection[W+i].first)
					front += 1
				end
			end
			
			new_layer = Layer{S,D,V}(Dict(collection[1:W-1-back]), false)
			@assert ((W-1-back) + length(Tie_set) + (length(layer)-W-front+1) == length(layer) )
			
			# merged_state, merged_node = merge([x.first for x in collection[W+front-1:end]], layer)
			if front + W == length(layer)
				state, node = collection[front+W]
				if !haskey(new_layer, state)
					new_layer[state] = node
				else				
					if new_layer[state].obj < node.obj
						new_layer[state] = node
					end				
				end
				
			elseif front + W < length(layer)	
				merged_state, merged_node = merge([x.first for x in collection[W+front:end]], layer)
		
				if !haskey(new_layer, merged_state)
					new_layer[merged_state] = merged_node
				else				
					if new_layer[merged_state].obj < merged_node.obj
						new_layer[merged_state] = merged_node
					end				
				end
			end			
			T_set = collect(Tie_set)
				
			merged_state, merged_node = merge(T_set, layer)
			if !haskey(new_layer, merged_state)
				new_layer[merged_state] = merged_node
			else				
				if new_layer[merged_state].obj < merged_node.obj
					new_layer[merged_state] = merged_node
				end				
			end			
			
		else
			
			new_layer = Layer{S,D,V}(Dict(collection[1:W-1]), false)			
			merged_state, merged_node = merge([x.first for x in collection[W:end]], layer)
			if !haskey(new_layer, merged_state)
				new_layer[merged_state] = merged_node
			else				
				if new_layer[merged_state].obj < merged_node.obj
					new_layer[merged_state] = merged_node
				end				
			end			
			
		end
		
	end
	
	return new_layer
	
end


### this function merges the last layer of the decision diagram into terminal state (node)

function last_layer_into_terminal(layer::Layer{S,D,V}) where {S,D,V}
	
	states = sort!(collect(layer), by=tup -> tup.second.obj, rev=true)		
	merged_state = Set([])
	return Layer{S,D,V}(Dict([merged_state => Node{S,D,V}(states[1].second.obj, states[1].second.inarc, false)]), false)
end


### this function merges the given states (nodes) into a single merged state (merged node)
function merge(states, layer::Layer{S,D,V}) where {S,D,V}
	
	sort!(states, by=tup -> layer[tup].obj, rev=true)
	node = layer.nodes[states[1]]
	merged_state = states[1]
	for state in states
		merged_state = Base.union(merged_state, state)
	end
	return merged_state, Node{S,D,V}(node.obj, node.inarc, false)
end

###################################

### this function collects the longest path of the decision diagram 

function longest_path(diagram::Diagram{S,D,V}) where {S,D,V}
    # Collect path in reverse, from terminal to root.
    terminal = only(values(diagram.layers[end].nodes))
    num_variables = length(diagram.partial_sol) + length(diagram.variables)
    decisions = Vector{D}(undef, num_variables)
    node, depth = terminal, length(diagram.layers) - 1
    while depth != 0
        decisions[diagram.variables[depth]] = node.inarc.decision
        state = node.inarc.tail
        node = diagram.layers[depth][state]
        depth -= 1
    end

    return Solution(decisions, terminal.obj)
end

function last_exact_layer(diagram)
    for (l, layer) in enumerate(diagram.layers)
        if !layer.exact
            # Current layer has at least one relaxed node.
            @assert l > 1

            # Return previous layer (all exact)
            return l - 1
        end
    end
    # If we reached the end then even the terminal layer is exact.
    return length(diagram.layers)
end
    

function branch_and_bound(instance; W_Primal, W_Dual, variable_ordering::AbstractString,  SELECTION::AbstractString)

	@assert (variable_ordering == "CDS" || variable_ordering == "MIN")
	
	@assert (SELECTION == "SO" || SELECTION == "BT")			
		
	state = initial_state(instance)
    S = typeof(state)
    D = domain_type(instance)
    V = value_type(instance)
    original_problem = Subproblem(Int[], D[], zero(V), state)
    problems = PriorityQueue(original_problem => zero(V))
    incumbent = Solution(D[], typemin(V))
	n_subproblems = 0
	
    # Solve subproblems, one at a time.
    while !isempty(problems)
        current = dequeue!(problems)
		n_subproblems += 1
        root_layer = Layer{S,D,V}(Dict(current.state => Node{S,D,V}(current.obj)))

        # solve primal
        diagram = Diagram{S,D,V}(current.variables, [root_layer], [])
		top_down_DD_primal!(diagram, instance, W=W_Primal)
		solution = longest_path(diagram)
		

        # update incumbent
        if solution.objective > incumbent.objective
            for (variable, decision) in zip(current.variables, current.decisions)
                solution.decisions[variable] = decision
            end
            incumbent = solution
        end

        # have we solved the subproblem already?
        if all(l -> l.exact, diagram.layers)
            continue
        end

        # solve dual 
        diagram = Diagram{S,D,V}(current.variables, [root_layer], []) 
		top_down_DD_dual!(diagram, instance; W=W_Dual, variable_order=variable_ordering, node_selection=SELECTION)
        solution = longest_path(diagram)
		
		
        # create subproblems if not pruned
        if solution.objective > incumbent.objective
            cutset = last_exact_layer(diagram)
            @assert length(diagram.layers[cutset]) > 1
            for (sub_state, sub_node) in diagram.layers[cutset]
                depth = cutset - 1
                new_decisions = Vector{D}(undef, depth)
                node = sub_node
                while depth != 0
                    new_decisions[depth] = node.inarc.decision
                    state = node.inarc.tail
                    node = diagram.layers[depth][state]
                    depth -= 1
                end

                variables = vcat(current.variables, diagram.variables[1:cutset - 1])
                decisions = vcat(current.decisions, new_decisions)

                subproblem = Subproblem(variables, decisions, sub_node.obj,                                                                              sub_state)
                problems[subproblem] = subproblem.obj
            end
        end
		
    end
	
    return incumbent, n_subproblems
end

#############################################################################
######  problem specific part

# In this part specification of every problem (i.e. format of an input (instance) of the problem, variable domain, initial state (root state), transition function) is written.

#########    include("MISP.jl")

module MISP

using ..DD
using ..DD: Arc, Node, Layer

struct Instance
    graph::Vector{Vector{Int64}}
    function Instance(graph)
        new(graph)
    end
end

Base.length(instance::Instance) = length(instance.graph) 
DD.domain_type(instance::Instance) = Bool
DD.value_type(instance::Instance) = Int64
 

DD.initial_state(instance::Instance) = Set([i for i in 1:length(instance)]) 


### the following function is an implementation of the transition function of the MISP's dynamic programming

function DD.transitions(instance::Instance, state, variable)

	results = Dict{Arc{typeof(state), Bool, Int64}, typeof(state)}()

	if !(isempty(state)) 
		if variable in state
			current = Set([variable])
			
			#true
			for i in state 
				if instance.graph[variable][i] == 0 
					push!(current, i)
				end
			end
			pop!(current, variable)
			results[Arc(state, true, 1)] = current
		
		else
			results[Arc(state, false, 0)] = state
		end
		
	else
		
		results[Arc(state, false, 0)] = state
		
	end
	
	if variable in state
		current_1 = Set([i for i in state])
		pop!(current_1, variable)
		results[Arc(state, false, 0)] = current_1
	end
    
    return results
end

end # module


### this function is for reading the graphs (generated randomly) and create an instance of MISP 

function read_random(size, density, id)

	instance = string("instance-",size, "-",  density, "-", id, ".dat")
	
	matrix = readdlm("address\\Random_Graphs\\Archive_1\\instance-$size\\dens-$density\\$instance")
	
	graph = Vector{Vector{Int64}}(undef, size)
	for i in 1:size
		graph[i] = Base.zeros(Int64, size)
	end
	
	for i in 1:Int(length(matrix)/2)
		node1 = Int(matrix[i,1]) + 1
		node2 = Int(matrix[i,2]) + 1
		graph[node1][node2] = 1
		graph[node2][node1] = 1
	end

	
	return DD.MISP.Instance(graph)
	
end



end



# ╔═╡ eb53789f-ec24-4a4b-849d-24547cd05ccc

###################     BB for MISP Random Graphs  #############


comment this line, then run the cell


begin
	
	Gsize = 100
	Gdensity = 0.4
	n_instance = 20
	MW_Primal = 100
	MW_Dual = 100
	orderings = ["MIN", "CDS"]
	selections = ["SO", "BT"]

	solution_times = Dict([]) 
	BB_nodes = Dict([])
	map = Dict([])
	
	counter = [1]
	for Node_Selection in selections
	
		for Variable_Ordering in orderings	
			
			map[counter[1]] = (Node_Selection, Variable_Ordering)
			
			solution_times[(Node_Selection, Variable_Ordering)] = []
			BB_nodes[(Node_Selection, Variable_Ordering)] = []			
			
			for instance in 1:n_instance

				problem_instance = DD.read_random(Gsize, Gdensity, instance)
				problem_diagram = DD.Diagram(problem_instance)

				run_time = Base.@elapsed solution, sub_size = DD.branch_and_bound(problem_instance; W_Primal=MW_Primal, W_Dual=MW_Dual, variable_ordering = Variable_Ordering, SELECTION = Node_Selection)

				value_opt = solution.objective

				Base.append!(solution_times[(Node_Selection, Variable_Ordering)], run_time)
				Base.append!(BB_nodes[(Node_Selection, Variable_Ordering)], sub_size)
				
			end	
			counter[1] += 1
		end
	end
	
	output_BB = DataFrame((

			combination = [map[i] for i in 1:length(map)],

			time = [sum(solution_times[map[i]])/n_instance for i in 1:length(map)],
			
			size = [sum(BB_nodes[map[i]])/n_instance for i in 1:length(map)]
				
		))

		CSV.write("desired address\\results_avg.csv", output_BB)
	
end


# ╔═╡ e8c6abcf-c83a-4594-be52-87251bf2885c

###################    Dual bounds for MISP Random Graphs  #############


comment this line, then run the cell


begin
	
	Gsize = 100
	Gdensity = 0.1
	n_instance = 20
	MW = 100
	orderings = ["MIN", "CDS"]
	selections = ["SO", "BT"]

	bounds = Dict([])
	times = Dict([])
	sizes = Dict([])
	
	map = Dict([])
	
	file_name = string("RG_Dual_$Gdensity", "_Size", Gsize, "(W", MW, ")")
	
	counter = [1]
	for NS in selections
	
		for VO in orderings	
			
			map[counter[1]] = (NS, VO)
			
			bounds[(NS,VO)] = []
			times[(NS,VO)] = []
			sizes[(NS,VO)] = []
			
			for instance in 1:n_instance

				problem_instance, opt = DD.read_random(Gsize, Gdensity, instance)
				problem_diagram = DD.Diagram(problem_instance)

				run_time = Base.@elapsed solution = DD.top_down_DD_dual!(problem_diagram, problem_instance; W=MW, variable_order = VO, node_selection = NS)

				DB = (solution-opt)/opt

				Base.append!(bounds[(NS, VO)], DB)
				Base.append!(times[(NS, VO)], run_time)
				Base.append!(sizes[(NS, VO)], sum([length(x) for x in problem_diagram.layers]))
						
				
				counter[1] += 1
			end	
			
		end
	end
	
	output_gaps = DataFrame((

			combination = [map[i] for i in 1:length(map)],

			Gap = [sum(bounds[map[i]])/n_instance for i in 1:length(map)],
			
			Time = [sum(times[map[i]])/n_instance for i in 1:length(map)],
			
			Size = [sum(sizes[map[i]])/n_instance for i in 1:length(map)]
							
		))

		CSV.write("address\\$Gdensity\\$file_name.csv", output_gaps)
	
end


# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
AutoHashEquals = "15f4f7f2-30c1-5605-9d31-71845cf9641f"
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
DataStructures = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
DelimitedFiles = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[compat]
AutoHashEquals = "~1.0.0"
CSV = "~0.10.15"
DataFrames = "~1.7.0"
DataStructures = "~0.18.20"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[AutoHashEquals]]
deps = ["Pkg"]
git-tree-sha1 = "7fc4d1532a3df01af51bae5c1d20389f5aeea086"
uuid = "15f4f7f2-30c1-5605-9d31-71845cf9641f"
version = "1.0.0"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

[[CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[Compat]]
deps = ["Dates", "LinearAlgebra", "TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"

[[Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "fb61b4812c49343d7ef0b533ba982c46021938a6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[FilePathsBase]]
deps = ["Compat", "Dates", "Mmap", "Test"]
git-tree-sha1 = "7878ff7172a8e6beedd1dea14bd27c3c6340d361"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.22"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[InlineStrings]]
deps = ["Parsers"]
git-tree-sha1 = "45521d31238e87ee9f9732561bfee12d4eebd52d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.2"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OrderedCollections]]
git-tree-sha1 = "12f1439c4f986bb868acda6ea33ebc78e19b95ad"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.7.0"

[[Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "66b20dd35966a748321d3b2537c4584cf40387c7"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.3.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a04cabe79c5f01f4d723cc6704070ada0b9d46d5"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.3.4"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╠═504a3e40-e0b5-11ef-0ef4-fb7ffa0af109
# ╠═eb53789f-ec24-4a4b-849d-24547cd05ccc
# ╠═e8c6abcf-c83a-4594-be52-87251bf2885c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
