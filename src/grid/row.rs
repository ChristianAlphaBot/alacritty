//! Defines the Row type which makes up lines in the grid.

use std::cmp::{max, min};
use std::iter::{self, Chain, RepeatN};
use std::ops::{Index, IndexMut, Range, RangeFrom, RangeFull, RangeTo, RangeToInclusive};
use std::{mem, ptr, slice};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::grid::GridCell;
use crate::index::Column;
use crate::term::cell::ResetDiscriminant;

/// A row in the grid.
///
/// cipher patch: a row in the scrollback history may be compact ([`Row::compact`]): it keeps its
/// width ([`Row::len`]), and the blank cells at its end ([`GridCell::blank`]) are not stored. A
/// cell read past what it stores is that blank ([`Index<Column>`], [`IntoIterator`]); any write,
/// and any mutable slice, stores its whole width again first ([`Row::expand`]). A shared slice
/// (`&row[..]` and the other ranges) covers only the cells it stores, so code that must see a
/// compact row's whole width reads it by column or iterates it.
#[derive(Default, Clone, Debug)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Row<T> {
    inner: Vec<T>,

    /// Maximum number of occupied entries.
    ///
    /// This is the upper bound on the number of elements in the row, which have been modified
    /// since the last reset. All cells after this point are guaranteed to be equal.
    pub(crate) occ: usize,

    /// cipher patch: how many cells at the row's end it does not store, each [`GridCell::blank`].
    /// Zero but for a compact history row ([`Row::compact`]).
    #[cfg_attr(feature = "serde", serde(default))]
    tail: usize,
}

impl<T: GridCell + PartialEq> PartialEq for Row<T> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.into_iter().eq(other)
    }
}

/// cipher patch: the least blank tail [`Row::compact`] drops. A shorter one stays stored, so a
/// row nearly full is not copied for a few bytes.
const COMPACT_LEAST: usize = 16;

impl<T: Default> Row<T> {
    /// Create a new terminal row.
    ///
    /// Ideally the `template` should be `Copy` in all performance sensitive scenarios.
    pub fn new(columns: usize) -> Row<T> {
        debug_assert!(columns >= 1);

        let mut inner: Vec<T> = Vec::with_capacity(columns);

        // This is a slightly optimized version of `std::vec::Vec::resize`.
        unsafe {
            let mut ptr = inner.as_mut_ptr();

            for _ in 1..columns {
                ptr::write(ptr, T::default());
                ptr = ptr.offset(1);
            }
            ptr::write(ptr, T::default());

            inner.set_len(columns);
        }

        Row { inner, occ: 0, tail: 0 }
    }

    /// Increase the number of columns in the row.
    #[inline]
    pub fn grow(&mut self, columns: usize)
    where
        T: GridCell,
    {
        self.expand();
        if self.inner.len() >= columns {
            return;
        }

        self.inner.resize_with(columns, T::default);
    }

    /// Reduce the number of columns in the row.
    ///
    /// This will return all non-empty cells that were removed.
    pub fn shrink(&mut self, columns: usize) -> Option<Vec<T>>
    where
        T: GridCell,
    {
        self.expand();
        if self.inner.len() <= columns {
            return None;
        }

        // Split off cells for a new row.
        let mut new_row = self.inner.split_off(columns);
        let index = new_row.iter().rposition(|c| !c.is_empty()).map_or(0, |i| i + 1);
        new_row.truncate(index);

        self.occ = min(self.occ, columns);

        if new_row.is_empty() { None } else { Some(new_row) }
    }

    /// Reset all cells in the row to the `template` cell.
    #[inline]
    pub fn reset<D>(&mut self, template: &T)
    where
        T: ResetDiscriminant<D> + GridCell,
        D: PartialEq,
    {
        self.expand();
        debug_assert!(!self.inner.is_empty());

        // Mark all cells as dirty if template cell changed.
        let len = self.inner.len();
        if self.inner[len - 1].discriminant() != template.discriminant() {
            self.occ = len;
        }

        // Reset every dirty cell in the row.
        for item in &mut self.inner[0..self.occ] {
            item.reset(template);
        }

        self.occ = 0;
    }
}

#[allow(clippy::len_without_is_empty)]
impl<T> Row<T> {
    #[inline]
    pub fn from_vec(vec: Vec<T>, occ: usize) -> Row<T> {
        Row { inner: vec, occ, tail: 0 }
    }

    /// The row's width: every cell it stores, and a compact row's blank tail.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len() + self.tail
    }

    /// cipher patch: how many of its cells the row stores ([`Row::compact`]).
    #[inline]
    pub fn stored(&self) -> usize {
        self.inner.len()
    }
}

impl<T: GridCell> Row<T> {
    /// cipher patch: drop the row's blank tail ([`GridCell::blank`]) from memory, keeping its
    /// width: every read of it is as before. For a row in the scrollback history, which nothing
    /// writes to; a write stores the tail again ([`Row::expand`]). A tail under [`COMPACT_LEAST`]
    /// cells stays.
    pub fn compact(&mut self)
    where
        T: PartialEq,
    {
        let Some(kept) = self.compactable() else { return };

        // A buffer of its own size, and the whole one freed.
        let mut cells = Vec::with_capacity(kept);
        cells.extend(self.inner.drain(..kept));
        self.tail += self.inner.len();
        self.inner = cells;
        // Every cell past what it stores is the blank.
        self.occ = min(self.occ, kept);
    }

    /// cipher patch: [`Row::compact`], handing this row's whole buffer, its cells as they were,
    /// to `next`: a compact row as wide, recycled from the oldest history for the terminal to
    /// reset and write to next. So a full history takes a line in without allocating a whole
    /// row. Its `occ` goes with the buffer, which its reset reads as upstream's reads any row's.
    pub fn compact_into(&mut self, next: &mut Row<T>)
    where
        T: PartialEq,
    {
        let Some(kept) = self.compactable() else { return };

        let cells = self.inner[..kept].to_vec();
        let whole = mem::replace(&mut self.inner, cells);
        let len = whole.len();
        if next.tail != 0 && next.len() == len {
            next.inner = whole;
            next.tail = 0;
            next.occ = self.occ;
        }
        self.tail = len - kept;
        self.occ = min(self.occ, kept);
    }

    /// The cells a compact row would keep, when it would drop at least [`COMPACT_LEAST`]: every
    /// cell up to its last that is not the blank, each read from its end.
    fn compactable(&self) -> Option<usize>
    where
        T: PartialEq,
    {
        if self.tail != 0 {
            return None;
        }
        let blank = T::blank();
        let kept = self.inner.iter().rposition(|cell| cell != blank).map_or(0, |last| last + 1);
        (self.inner.len() - kept >= COMPACT_LEAST).then_some(kept)
    }

    /// cipher patch: store a compact row's blank tail again, so every cell of it can be written.
    #[inline]
    pub fn expand(&mut self) {
        if self.tail == 0 {
            return;
        }

        let len = self.len();
        self.inner.resize_with(len, || T::blank().clone());
        self.tail = 0;
    }

    #[inline]
    pub fn last(&self) -> Option<&T> {
        if self.tail != 0 { Some(T::blank()) } else { self.inner.last() }
    }

    #[inline]
    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.expand();
        self.occ = self.inner.len();
        self.inner.last_mut()
    }

    #[inline]
    pub fn append(&mut self, vec: &mut Vec<T>) {
        self.expand();
        self.occ += vec.len();
        self.inner.append(vec);
    }

    #[inline]
    pub fn append_front(&mut self, mut vec: Vec<T>) {
        self.expand();
        self.occ += vec.len();

        vec.append(&mut self.inner);
        self.inner = vec;
    }

    /// Check if all cells in the row are empty.
    #[inline]
    pub fn is_clear(&self) -> bool {
        self.inner.iter().all(GridCell::is_empty) && (self.tail == 0 || T::blank().is_empty())
    }

    #[inline]
    pub fn front_split_off(&mut self, at: usize) -> Vec<T> {
        self.expand();
        self.occ = self.occ.saturating_sub(at);

        let mut split = self.inner.split_off(at);
        std::mem::swap(&mut split, &mut self.inner);
        split
    }
}

impl<'a, T: GridCell> IntoIterator for &'a Row<T> {
    type IntoIter = Chain<slice::Iter<'a, T>, RepeatN<&'a T>>;
    type Item = &'a T;

    /// Every cell of the row, a compact row's blank tail too.
    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.inner.iter().chain(iter::repeat_n(T::blank(), self.tail))
    }
}

impl<'a, T: GridCell> IntoIterator for &'a mut Row<T> {
    type IntoIter = slice::IterMut<'a, T>;
    type Item = &'a mut T;

    #[inline]
    fn into_iter(self) -> slice::IterMut<'a, T> {
        self.expand();
        self.occ = self.len();
        self.inner.iter_mut()
    }
}

impl<T: GridCell> Index<Column> for Row<T> {
    type Output = T;

    #[inline]
    fn index(&self, index: Column) -> &T {
        match self.inner.get(index.0) {
            Some(cell) => cell,
            // A compact row's blank tail; past its width, a panic as an index out of bounds was.
            None if index.0 < self.len() => T::blank(),
            None => out_of_bounds(index.0, self.len()),
        }
    }
}

#[cold]
#[inline(never)]
fn out_of_bounds(column: usize, len: usize) -> ! {
    panic!("column {column} of a row {len} wide")
}

impl<T: GridCell> IndexMut<Column> for Row<T> {
    #[inline]
    fn index_mut(&mut self, index: Column) -> &mut T {
        if index.0 >= self.inner.len() {
            self.expand();
        }
        self.occ = max(self.occ, *index + 1);
        &mut self.inner[index.0]
    }
}

impl<T> Index<Range<Column>> for Row<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: Range<Column>) -> &[T] {
        &self.inner[(index.start.0)..(index.end.0)]
    }
}

impl<T: GridCell> IndexMut<Range<Column>> for Row<T> {
    #[inline]
    fn index_mut(&mut self, index: Range<Column>) -> &mut [T] {
        self.expand();
        self.occ = max(self.occ, *index.end);
        &mut self.inner[(index.start.0)..(index.end.0)]
    }
}

impl<T> Index<RangeTo<Column>> for Row<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: RangeTo<Column>) -> &[T] {
        &self.inner[..(index.end.0)]
    }
}

impl<T: GridCell> IndexMut<RangeTo<Column>> for Row<T> {
    #[inline]
    fn index_mut(&mut self, index: RangeTo<Column>) -> &mut [T] {
        self.expand();
        self.occ = max(self.occ, *index.end);
        &mut self.inner[..(index.end.0)]
    }
}

impl<T> Index<RangeFrom<Column>> for Row<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: RangeFrom<Column>) -> &[T] {
        &self.inner[(index.start.0)..]
    }
}

impl<T: GridCell> IndexMut<RangeFrom<Column>> for Row<T> {
    #[inline]
    fn index_mut(&mut self, index: RangeFrom<Column>) -> &mut [T] {
        self.expand();
        self.occ = self.len();
        &mut self.inner[(index.start.0)..]
    }
}

impl<T> Index<RangeFull> for Row<T> {
    type Output = [T];

    #[inline]
    fn index(&self, _: RangeFull) -> &[T] {
        &self.inner[..]
    }
}

impl<T: GridCell> IndexMut<RangeFull> for Row<T> {
    #[inline]
    fn index_mut(&mut self, _: RangeFull) -> &mut [T] {
        self.expand();
        self.occ = self.len();
        &mut self.inner[..]
    }
}

impl<T> Index<RangeToInclusive<Column>> for Row<T> {
    type Output = [T];

    #[inline]
    fn index(&self, index: RangeToInclusive<Column>) -> &[T] {
        &self.inner[..=(index.end.0)]
    }
}

impl<T: GridCell> IndexMut<RangeToInclusive<Column>> for Row<T> {
    #[inline]
    fn index_mut(&mut self, index: RangeToInclusive<Column>) -> &mut [T] {
        self.expand();
        self.occ = max(self.occ, *index.end + 1);
        &mut self.inner[..=(index.end.0)]
    }
}
