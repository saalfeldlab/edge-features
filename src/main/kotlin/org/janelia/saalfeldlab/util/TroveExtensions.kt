package org.janelia.saalfeldlab.util

import gnu.trove.map.TLongObjectMap


public inline fun <T> TLongObjectMap<T>.computeIfAbsent(key: Long, mappingFunction: (Long) -> T) = this[key] ?: putAndReturn(key, mappingFunction(key))

public fun <T> TLongObjectMap<T>.putAndReturn(key: Long, value: T): T {
    this.put(key, value)
    return value
}
