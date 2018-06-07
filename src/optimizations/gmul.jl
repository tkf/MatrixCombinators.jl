function _mul!(executor::DefaultExecutor, Y, M::AddedMatrices, X)
    # TODO: It would be nice if the "execution strategy" is decided at
    # the time AddedMatrices is constructed and then embedded in the
    # type (parameter).  But probably there is not so much gain here
    # since the matrices are assumed to be big anyway.
    if has_gmul(Y, M.B, X) && has_mul(Y, M.A, X)
        mul!(Y, M.A, X)
        amul!(Y, M.B, X)
    elseif has_gmul(Y, M.A, X) && has_mul(Y, M.B, X)
        mul!(Y, M.B, X)
        amul!(Y, M.A, X)
    else
        dumb_exr = DumbExecutor(executor.allocator)
        _mul!(dumb_exr, Y, M, X)
    end
    return Y
end
