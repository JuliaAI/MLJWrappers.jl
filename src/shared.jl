prepend(s::Symbol, ::Nothing) = s
prepend(s::Symbol, t::Symbol) = Expr(:(.), s, QuoteNode(t))
prepend(s::Symbol, ex::Expr) = Expr(:(.), prepend(s, ex.args[1]), ex.args[2])
