// Copyright 2022 the Kurbo Authors
// SPDX-License-Identifier: Apache-2.0 OR MIT

//! Simplification of a Bézier path.
//!
//! This module is currently experimental.
//!
//! The methods in this module create a `SimplifyBezPath` object, which can then
//! be fed to [`fit_to_bezpath`] or [`fit_to_bezpath_opt`] depending on the degree
//! of optimization desired.
//!
//! The implementation uses a number of techniques to achieve high performance and
//! accuracy. The parameter (generally written `t`) evenly divides the curve segments
//! in the original, so sampling can be done in constant time. The derivatives are
//! computed analytically, as that is straightforward with Béziers.
//!
//! The areas and moments are computed analytically (using Green's theorem), and
//! the prefix sum is stored. Thus, it is possible to analytically compute the area
//! and moment of any subdivision of the curve, also in constant time, by taking
//! the difference of two stored prefix sum values, then fixing up the subsegments.
//!
//! A current limitation (hoped to be addressed in the future) is that non-regular
//! cubic segments may have tangents computed incorrectly. This can easily happen,
//! for example when setting a control point equal to an endpoint.
//!
//! In addition, this method does not report corners (adjoining segments where the
//! tangents are not continuous). It is not clear whether it's best to handle such
//! cases here, or in client code.
//!
//! [`fit_to_bezpath`]: crate::fit_to_bezpath
//! [`fit_to_bezpath_opt`]: crate::fit_to_bezpath_opt

use alloc::vec::Vec;

use core::ops::Range;

#[cfg(not(feature = "std"))]
use crate::common::FloatFuncs;

use crate::{
    BezPath, CubicBez, CurveFitSample, Line, ParamCurve, ParamCurveDeriv, ParamCurveFit, PathEl,
    PathSeg, Point, QuadBez, Vec2, fit_to_bezpath, fit_to_bezpath_opt,
};

/// A Bézier path which has been prepared for simplification.
///
/// See the [module-level documentation] for a bit more discussion of the approach,
/// and how this struct is to be used.
///
/// [module-level documentation]: crate::simplify
pub struct SimplifyBezPath {
    els: Vec<SimplifyCubic>,
    /// The reference point the moment integrals are computed relative to.
    ///
    /// The moments are accumulated relative to this point rather than in
    /// absolute coordinates, so that a path far from the origin doesn't lose
    /// most of its precision to cancellation.
    origin: Point,
}

struct SimplifyCubic {
    c: CubicBez,
    // The inclusive prefix sum of the moment integrals, relative to `origin`
    moments: (f64, f64, f64),
}

/// Options for simplifying paths.
pub struct SimplifyOptions {
    /// The tangent of the minimum angle below which the path is considered smooth.
    angle_thresh: f64,
    opt_level: SimplifyOptLevel,
}

/// Optimization level for simplification.
pub enum SimplifyOptLevel {
    /// Subdivide; faster but not as optimized results.
    Subdivide,
    /// Optimize subdivision points.
    Optimize,
}

impl Default for SimplifyOptions {
    fn default() -> Self {
        let opt_level = SimplifyOptLevel::Subdivide;
        SimplifyOptions {
            angle_thresh: 1e-3,
            opt_level,
        }
    }
}

#[doc(hidden)]
/// Compute moment integrals.
///
/// This is exposed for testing purposes but is an internal detail. We can
/// add to the public, documented interface if there is a use case.
pub fn moment_integrals(c: CubicBez) -> (f64, f64, f64) {
    let (x0, y0) = (c.p0.x, c.p0.y);
    let (x1, y1) = (c.p1.x - x0, c.p1.y - y0);
    let (x2, y2) = (c.p2.x - x0, c.p2.y - y0);
    let (x3, y3) = (c.p3.x - x0, c.p3.y - y0);

    let r0 = 3. * x1;
    let r1 = 3. * y1;
    let r2 = x2 * y3;
    let r3 = x3 * y2;
    let r4 = x3 * y3;
    let r5 = 27. * y1;
    let r6 = x1 * x2;
    let r7 = 27. * y2;
    let r8 = 45. * r2;
    let r9 = 18. * x3;
    let r10 = x1 * y1;
    let r11 = 30. * x1;
    let r12 = 45. * x3;
    let r13 = x2 * y1;
    let r14 = 45. * r3;
    let r15 = x1.powi(2);
    let r16 = 18. * y3;
    let r17 = x2.powi(2);
    let r18 = 45. * y3;
    let r19 = x3.powi(2);
    let r20 = 30. * y1;
    let r21 = y2.powi(2);
    let r22 = y3.powi(2);
    let r23 = y1.powi(2);
    let a = -r0 * y2 - r0 * y3 + r1 * x2 + r1 * x3 - 6. * r2 + 6. * r3 + 10. * r4;

    // Scale and add chord
    let lift = x3 * y0;
    let area = a * 0.05 + lift;
    let x = r10 * r9 - r11 * r4 + r12 * r13 + r14 * x2 - r15 * r16 - r15 * r7 - r17 * r18
        + r17 * r5
        + r19 * r20
        + 105. * r19 * y2
        + 280. * r19 * y3
        - 105. * r2 * x3
        + r5 * r6
        - r6 * r7
        - r8 * x1;
    let y = -r10 * r16 - r10 * r7 - r11 * r22 + r12 * r21 + r13 * r7 + r14 * y1 - r18 * x1 * y2
        + r20 * r4
        - 27. * r21 * x1
        - 105. * r22 * x2
        + 140. * r22 * x3
        + r23 * r9
        + 27. * r23 * x2
        + 105. * r3 * y3
        - r8 * y2;

    let mx = x * (1. / 840.) + x0 * area + 0.5 * x3 * lift;
    let my = y * (1. / 420.) + y0 * a * 0.1 + y0 * lift;

    (area, mx, my)
}

impl SimplifyBezPath {
    /// Set up a new Bézier path for simplification.
    ///
    /// Currently this is not dealing with discontinuities at all, but it
    /// could be extended to do so.
    pub fn new(path: impl IntoIterator<Item = PathEl>) -> Self {
        let mut origin = Point::ORIGIN;
        let (mut a, mut x, mut y) = (0.0, 0.0, 0.0);
        let mut els: Vec<SimplifyCubic> = Vec::new();
        for seg in crate::segments(path) {
            let c = seg.to_cubic();
            if els.is_empty() {
                origin = c.p0;
            }
            let (ai, xi, yi) = moment_integrals(translate(c, origin));
            a += ai;
            x += xi;
            y += yi;
            els.push(SimplifyCubic {
                c,
                moments: (a, x, y),
            });
        }
        SimplifyBezPath { els, origin }
    }

    /// Resolve a `t` value to a cubic.
    ///
    /// Also return the resulting `t` value for the selected cubic.
    fn scale(&self, t: f64) -> (usize, f64) {
        let t_scale = t * self.els.len() as f64;
        let t_floor = t_scale.floor();
        (t_floor as usize, t_scale - t_floor)
    }

    /// Moment integrals of a subsegment, relative to `self.origin`.
    fn moment_integrals(&self, i: usize, range: Range<f64>) -> (f64, f64, f64) {
        if range.end == range.start {
            (0.0, 0.0, 0.0)
        } else {
            moment_integrals(translate(self.els[i].c.subsegment(range), self.origin))
        }
    }

    /// Evaluate the curve, clamping the parameter to the valid range.
    fn eval(&self, t: f64) -> Point {
        let (mut i, mut t0) = self.scale(t);
        if i == self.els.len() {
            i -= 1;
            t0 = 1.0;
        }
        self.els[i].c.eval(t0)
    }
}

/// Translate a cubic so that `origin` is at the origin.
fn translate(c: CubicBez, origin: Point) -> CubicBez {
    let v = origin.to_vec2();
    CubicBez::new(c.p0 - v, c.p1 - v, c.p2 - v, c.p3 - v)
}

impl ParamCurveFit for SimplifyBezPath {
    fn sample_pt_deriv(&self, t: f64) -> (Point, Vec2) {
        let (mut i, mut t0) = self.scale(t);
        let n = self.els.len();
        if i == n {
            i -= 1;
            t0 = 1.0;
        }
        let c = self.els[i].c;
        (c.eval(t0), c.deriv().eval(t0).to_vec2() * n as f64)
    }

    fn sample_pt_tangent(&self, t: f64, _: f64) -> CurveFitSample {
        let (mut i, mut t0) = self.scale(t);
        if i == self.els.len() {
            i -= 1;
            t0 = 1.0;
        }
        let c = self.els[i].c;
        let p = c.eval(t0);
        let tangent = c.deriv().eval(t0).to_vec2();
        CurveFitSample { p, tangent }
    }

    // We could use the default implementation, but provide our own, mostly
    // because it is possible to efficiently provide an analytically accurate
    // answer.
    fn moment_integrals(&self, range: Range<f64>) -> (f64, f64, f64) {
        let (i0, t0) = self.scale(range.start);
        let (i1, t1) = self.scale(range.end);
        // These are relative to `self.origin`.
        let (a, x, y) = if i0 == i1 {
            self.moment_integrals(i0, t0..t1)
        } else {
            let (a0, x0, y0) = self.moment_integrals(i0, t0..1.0);
            let (a1, x1, y1) = self.moment_integrals(i1, 0.0..t1);
            let (mut a, mut x, mut y) = (a0 + a1, x0 + x1, y0 + y1);
            if i1 > i0 + 1 {
                let (a2, x2, y2) = self.els[i0].moments;
                let (a3, x3, y3) = self.els[i1 - 1].moments;
                a += a3 - a2;
                x += x3 - x2;
                y += y3 - y2;
            }
            (a, x, y)
        };
        // Shift the reference point from `self.origin` to the start of the
        // range, as the trait requires. With `u = x - origin.x` and
        // `v = y - origin.y`, and `s` the start point relative to `origin`:
        //
        // ∫(v - s.y) dx           = a - s.y ∫dx
        // ∫(u - s.x)(v - s.y) dx  = x - s.x a - s.y ∫u dx + s.x s.y ∫dx
        // ∫(v - s.y)² dx          = y - 2 s.y a + s.y² ∫dx
        //
        // where `∫dx` and `∫u dx` over the range depend only on its endpoints.
        let s = self.eval(range.start) - self.origin;
        let u_end = self.eval(range.end).x - self.origin.x;
        let int_dx = u_end - s.x;
        let int_u_dx = 0.5 * (u_end + s.x) * int_dx;
        let x = x - s.x * a - s.y * int_u_dx + s.x * s.y * int_dx;
        let y = y - 2.0 * s.y * a + s.y * s.y * int_dx;
        let a = a - s.y * int_dx;
        (a, x, y)
    }

    fn break_cusp(&self, _: Range<f64>) -> Option<f64> {
        None
    }
}

#[derive(Default)]
struct SimplifyState {
    queue: BezPath,
    result: BezPath,
    needs_moveto: bool,
}

impl SimplifyState {
    fn add_seg(&mut self, seg: PathSeg) {
        if self.queue.is_empty() {
            self.queue.move_to(seg.start());
        }
        match seg {
            PathSeg::Line(l) => self.queue.line_to(l.p1),
            PathSeg::Quad(q) => self.queue.quad_to(q.p1, q.p2),
            PathSeg::Cubic(c) => self.queue.curve_to(c.p1, c.p2, c.p3),
        }
    }

    fn flush(&mut self, accuracy: f64, options: &SimplifyOptions) {
        if self.queue.is_empty() {
            return;
        }
        if self.queue.elements().len() == 2 {
            // Queue is just one segment (count is moveto + primitive)
            // Just output the segment, no simplification is possible.
            self.result
                .extend(self.queue.iter().skip(!self.needs_moveto as usize));
        } else {
            let s = SimplifyBezPath::new(&self.queue);
            let b = match options.opt_level {
                SimplifyOptLevel::Subdivide => fit_to_bezpath(&s, accuracy),
                SimplifyOptLevel::Optimize => fit_to_bezpath_opt(&s, accuracy),
            };
            self.result
                .extend(b.iter().skip(!self.needs_moveto as usize));
        }
        self.needs_moveto = false;
        self.queue.truncate(0);
    }
}

/// Simplify a Bézier path.
///
/// This function simplifies an arbitrary Bézier path; it is designed to handle
/// multiple subpaths and also corners.
///
/// The underlying curve-fitting approach works best if the source path is very
/// smooth. If it contains higher frequency noise, then results may be poor, as
/// the resulting curve matches the original with G1 continuity at each subdivision
/// point, and also preserves the area. For such inputs, consider some form of
/// smoothing or low-pass filtering before simplification. In particular, if the
/// input is derived from a sequence of points, consider fitting a smooth spline.
///
/// We may add such capabilities in the future, possibly as opt-in smoothing
/// specified through the options.
pub fn simplify_bezpath(
    path: impl IntoIterator<Item = PathEl>,
    accuracy: f64,
    options: &SimplifyOptions,
) -> BezPath {
    let mut last_pt = None;
    let mut last_seg: Option<PathSeg> = None;
    let mut state = SimplifyState::default();
    for el in path {
        let mut this_seg = None;
        match el {
            PathEl::MoveTo(p) => {
                state.flush(accuracy, options);
                state.needs_moveto = true;
                last_pt = Some(p);
            }
            PathEl::LineTo(p) => {
                let last = last_pt.unwrap();
                if last == p {
                    continue;
                }
                this_seg = Some(PathSeg::Line(Line::new(last, p)));
            }
            PathEl::QuadTo(p1, p2) => {
                let last = last_pt.unwrap();
                if last == p1 && last == p2 {
                    continue;
                }
                this_seg = Some(PathSeg::Quad(QuadBez::new(last, p1, p2)));
            }
            PathEl::CurveTo(p1, p2, p3) => {
                let last = last_pt.unwrap();
                if last == p1 && last == p2 && last == p3 {
                    continue;
                }
                this_seg = Some(PathSeg::Cubic(CubicBez::new(last, p1, p2, p3)));
            }
            PathEl::ClosePath => {
                state.flush(accuracy, options);
                state.result.close_path();
                state.needs_moveto = true;
                last_seg = None;
                continue;
            }
        }
        if let Some(seg) = this_seg {
            if let Some(last) = last_seg {
                let last_tan = last.tangents().1;
                let this_tan = seg.tangents().0;
                let cross = last_tan.cross(this_tan);
                let dot = last_tan.dot(this_tan);
                // The `|sin θ| > |cos θ| * thresh` test is periodic with period π,
                // so on its own it cannot tell a fold from a straight join; a 180°
                // reversal scores as well as no turn at all. A negative dot product
                // means the turn is more than a right angle, which is always a
                // corner, so test that separately.
                if dot <= 0.0 || cross.abs() > dot * options.angle_thresh {
                    state.flush(accuracy, options);
                }
            }
            last_pt = Some(seg.end());
            state.add_seg(seg);
        }
        last_seg = this_seg;
    }
    state.flush(accuracy, options);
    state.result
}

impl SimplifyOptions {
    /// Set optimization level.
    pub fn opt_level(mut self, level: SimplifyOptLevel) -> Self {
        self.opt_level = level;
        self
    }

    /// Set angle threshold.
    ///
    /// The tangent of the angle below which joins are considered smooth and
    /// not corners. The default is approximately 1 milliradian. A join turning
    /// by more than a right angle is always considered a corner, whatever the
    /// threshold.
    pub fn angle_thresh(mut self, thresh: f64) -> Self {
        self.angle_thresh = thresh;
        self
    }
}

#[cfg(test)]
mod tests {
    use crate::{BezPath, ParamCurve, ParamCurveFit, PathEl, Point, Shape, Vec2};

    use super::{SimplifyBezPath, SimplifyOptLevel, SimplifyOptions, simplify_bezpath};

    #[test]
    fn simplify_lines_corner() {
        // Make sure lines are passed through unchanged if there is a corner.
        let mut path = BezPath::new();
        path.move_to((1., 2.));
        path.line_to((3., 4.));
        path.line_to((10., 5.));
        let options = SimplifyOptions::default();
        let simplified = simplify_bezpath(path.clone(), 1.0, &options);
        assert_eq!(path, simplified);
    }

    /// All points of a path, control points included.
    fn all_points(path: &BezPath) -> Vec<Point> {
        path.elements()
            .iter()
            .flat_map(|el| match *el {
                PathEl::MoveTo(p) | PathEl::LineTo(p) => vec![p],
                PathEl::QuadTo(p1, p2) => vec![p1, p2],
                PathEl::CurveTo(p1, p2, p3) => vec![p1, p2, p3],
                PathEl::ClosePath => vec![],
            })
            .collect()
    }

    // A shallow curve arriving at (493.93, 324), then a straight backtrack
    // along y = 324. Fitting across that 180 degree fold used to produce a
    // control point tens of thousands of units away, which the fitter's own
    // error metric reported as an essentially exact fit.
    //
    // See https://github.com/linebender/kurbo/issues/604
    const FOLD_PTS: &[(f64, f64)] = &[
        (486.53000000000003, 325.49),
        (488.93, 324.68),
        (491.41, 324.18),
        (493.93, 324.0),
        (415.74, 324.0),
    ];

    fn fold_path() -> BezPath {
        let mut path = BezPath::new();
        path.move_to(Point::new(FOLD_PTS[0].0, FOLD_PTS[0].1));
        for (x, y) in &FOLD_PTS[1..] {
            path.line_to(Point::new(*x, *y));
        }
        path
    }

    /// Fit the fold path directly, bypassing corner detection.
    ///
    /// Even when a fold is handed to the fitter, it must not accept a curve
    /// that wanders far outside the source.
    #[test]
    fn fit_opt_fold_no_control_point_excursion() {
        let path = fold_path();
        let accuracy = 2.0;
        let bbox = path.bounding_box().inflate(4.0 * accuracy, 4.0 * accuracy);
        let s = SimplifyBezPath::new(&path);
        let fitted = crate::fit_to_bezpath_opt(&s, accuracy);
        for p in all_points(&fitted) {
            assert!(
                bbox.contains(p),
                "fitted path point {p:?} is outside {bbox:?}; path is {}",
                fitted.to_svg()
            );
        }
    }

    /// The quantities the fit is derived from are documented as being invariant
    /// to translation; make sure the arithmetic actually is.
    ///
    /// Computing the moment integrals in absolute coordinates loses about 14 of
    /// f64's ~16 digits to cancellation for a path this far from the origin,
    /// leaving the control arm lengths derived from them meaningless.
    #[test]
    fn moment_integrals_translation_invariance() {
        let path = fold_path();
        // The same shape, translated so it sits near the origin.
        let offset = Vec2::new(-400.0, -324.0);
        let translated = crate::Affine::translate(offset) * path.clone();

        let s = SimplifyBezPath::new(&path);
        let s_translated = SimplifyBezPath::new(&translated);
        for (t0, t1) in [
            (0.0, 1.0),
            (0.0, 0.5),
            (0.6, 1.0),
            (0.75, 1.0),
            (0.1, 0.2),
            (0.45, 0.55),
        ] {
            let m = ParamCurveFit::moment_integrals(&s, t0..t1);
            let m_translated = ParamCurveFit::moment_integrals(&s_translated, t0..t1);
            for (v, w) in [
                (m.0, m_translated.0),
                (m.1, m_translated.1),
                (m.2, m_translated.2),
            ] {
                // The paths differ by about 1e-14 relative, as translating them
                // rounds; allow a good deal more than that but far less than
                // the loss from cancellation.
                let tol = 1e-9 * v.abs().max(w.abs()).max(1e-6);
                assert!(
                    (v - w).abs() <= tol,
                    "moment integrals over {t0}..{t1} are not translation invariant: \
                     {m:?} vs {m_translated:?}"
                );
            }
        }
    }

    /// The fit itself should not change materially when the path is translated.
    #[test]
    fn fit_opt_translation_invariance() {
        let accuracy = 2.0;
        let path = fold_path();
        let offset = Vec2::new(-400.0, -324.0);
        let translated = crate::Affine::translate(offset) * path.clone();

        let fitted = crate::fit_to_bezpath_opt(&SimplifyBezPath::new(&path), accuracy);
        let fitted_translated =
            crate::fit_to_bezpath_opt(&SimplifyBezPath::new(&translated), accuracy);

        // Compare the two as shapes: the control points, the subdivision points
        // and even the number of segments are free to differ a little.
        let shifted = crate::Affine::translate(offset) * fitted.clone();
        let d = max_deviation(&shifted, &fitted_translated)
            .max(max_deviation(&fitted_translated, &shifted));
        assert!(
            d < accuracy,
            "fit differs by translation by {d}: {} vs {}",
            fitted.to_svg(),
            fitted_translated.to_svg()
        );
    }

    /// An approximate one-sided Hausdorff distance from `a` to `b`.
    fn max_deviation(a: &BezPath, b: &BezPath) -> f64 {
        use crate::ParamCurveNearest;
        const N: usize = 100;
        let segs: Vec<_> = b.segments().collect();
        let mut max_dist2: f64 = 0.0;
        for seg in a.segments() {
            for i in 0..=N {
                let p = seg.eval(i as f64 / N as f64);
                let dist2 = segs
                    .iter()
                    .map(|s| s.nearest(p, 1e-9).distance_sq)
                    .fold(f64::INFINITY, f64::min);
                max_dist2 = max_dist2.max(dist2);
            }
        }
        max_dist2.sqrt()
    }

    #[test]
    fn simplify_fold_no_control_point_excursion() {
        let path = fold_path();
        let accuracy = 2.0;
        let bbox = path.bounding_box().inflate(4.0 * accuracy, 4.0 * accuracy);
        for opt_level in [SimplifyOptLevel::Subdivide, SimplifyOptLevel::Optimize] {
            let options = SimplifyOptions::default()
                .angle_thresh(0.25)
                .opt_level(opt_level);
            let simplified = simplify_bezpath(path.iter(), accuracy, &options);
            for p in all_points(&simplified) {
                assert!(
                    bbox.contains(p),
                    "simplified path point {p:?} is outside {bbox:?}; path is {}",
                    simplified.to_svg()
                );
            }
        }
    }
}
