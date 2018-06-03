function _mul!(executor::DefaultExecutor, Y, M::AddedMatrices, X)
    if has_gmul(Y, M.B, X)
        mul!(Y, M.A, X)
        gmul!(Y, M.B, X)
    elseif has_gmul(Y, M.A, X)
        mul!(Y, M.B, X)
        gmul!(Y, M.A, X)
    else
        dumb_exr = DumbExecutor(executor.allocator)
        _mul!(dumb_exr, Y, M, X)
    end
    return Y
end
