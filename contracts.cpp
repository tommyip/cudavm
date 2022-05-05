#include "contracts.h"

// Constant Product AMM
//
// Accounts:
// 0. Pool source reserve
// 1. Pool destination reserve
// 2. Source wallet
// 3. Destination wallet
//
// Parameters:
// 0. Amount of in token to swap
//
// Equations:
//
// Constant Product:
// x * y = k  [x => pool source reserve, y => pool destination reserve]
//
// Constant Product Swap:
// (x + u) * (y - v) = k  [u => deposit, v => withdrawal]
//
// Withdrawal amount:
// v = y * u / (x + u)
Program constant_swap() {
    Program prog;

    // Calculate v
    // Load y: pool destination reserve
    prog.Const(1); prog.Const(0); prog.Load();
    // Load u: argument0 / deposit amount
    prog.Const(0); prog.Arg();
    // y * u
    prog.Mul();
    // Load x: pool source reserve
    prog.Const(0); prog.Const(0); prog.Load();
    // Load u
    prog.Const(0); prog.Arg();
    // x + u
    prog.Add();
    // v = y * u / (x + u)
    prog.Div();

    // Debit pool destination reserve
    prog.Dup();
    prog.Const(1); prog.Const(0); prog.Load();
    prog.Rot();
    prog.Sub();
    // Check resulting reserve >= 0
    prog.Dup(); prog.Const(-1); prog.Rot(); prog.Lt(); prog.Assert();
    prog.Const(1); prog.Const(0); prog.Store();

    // Credit destination wallet
    prog.Const(3); prog.Const(0); prog.Load();
    prog.Add();
    prog.Const(3); prog.Const(0); prog.Store();

    // Credit pool source reserve
    prog.Const(0); prog.Const(0); prog.Load();
    prog.Const(0); prog.Arg();
    prog.Add();
    prog.Const(0); prog.Const(0); prog.Store();

    // Debit source wallet
    prog.Const(2); prog.Const(0); prog.Load();
    prog.Const(0); prog.Arg();
    prog.Sub();
    // Check resulting balance >= 0
    prog.Dup(); prog.Const(-1); prog.Rot(); prog.Lt(); prog.Assert();
    prog.Const(2); prog.Const(0); prog.Store();

    return prog;
}

// User to user payment
//
// Accounts:
// 0. Payer wallet
// 1. Payee wallet
//
// Parameters:
// 0. Amount to tokens to transfer
Program payment() {
    Program prog;

    // Load payer balance
    prog.Const(0); prog.Const(0); prog.Load();
    // Load transfer amount
    prog.Const(0); prog.Arg();

    // Check resulting payer balance is positive
    prog.Sub(); prog.Dup(); prog.Const(0); prog.Lt(); prog.Not(); prog.Assert();

    // Debit payer
    prog.Const(0); prog.Const(0); prog.Store();

    // Credit payee
    prog.Const(1); prog.Const(0); prog.Load();
    prog.Const(0); prog.Arg();
    prog.Add();
    prog.Const(1); prog.Const(0); prog.Store();

    return prog;
}

Program oracle() {
    // TODO
    Program prog;
    return prog;
}
